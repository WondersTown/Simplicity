import asyncio
import atexit
import queue
import threading
import time
from typing import AsyncGenerator, Optional

import streamlit as st
from stone_brick.llm import (
    EndResult,
    EventTaskOutput,
    EventTaskOutputStream,
    EventTaskOutputStreamDelta,
    TaskEventDeps,
)

from simplicity.engines.pardo.engine import PardoEngine
from simplicity.resources import Resource
from simplicity.utils import get_settings_from_project_root

# import logfire
# logfire.configure()
# logfire.instrument_pydantic_ai()
# logfire.instrument_httpx()


class AsyncLoopManager:
    """Manages a single async event loop for the entire Streamlit app"""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._shutdown_event = threading.Event()

    def get_loop(self):
        """Get or create the async event loop"""
        if self._loop is None or self._loop.is_closed():
            self._start_loop()
        return self._loop

    def _start_loop(self):
        """Start the async event loop in a separate thread"""
        if self._thread is not None and self._thread.is_alive():
            return

        loop_ready = threading.Event()

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop_ready.set()

            try:
                # Run until shutdown is requested
                while not self._shutdown_event.is_set():
                    try:
                        self._loop.run_until_complete(asyncio.sleep(0.5))
                    except Exception as e:
                        # Log the exception but don't break immediately
                        print(f"AsyncLoopManager exception: {e}")
                        time.sleep(1.0)
                        continue
            finally:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()

                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                finally:
                    self._loop.close()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for the loop to be ready
        if not loop_ready.wait(timeout=10.0):
            raise RuntimeError("Failed to start event loop within timeout")

    def run_async_generator(self, async_gen_func, result_queue):
        """Run an async generator and put results in a queue"""
        loop = self.get_loop()
        if loop is None or loop.is_closed():
            raise RuntimeError("Event loop is not available")

        async def collect_results():
            try:
                async for item in async_gen_func():
                    result_queue.put(("data", item))
            except Exception as e:
                result_queue.put(("error", e))
            finally:
                result_queue.put(("done", None))

        try:
            _future = asyncio.run_coroutine_threadsafe(collect_results(), loop)
            # Don't wait for completion here, let the main thread handle it
        except Exception as e:
            result_queue.put(("error", e))
            result_queue.put(("done", None))

    def shutdown(self):
        """Shutdown the event loop and thread"""
        if self._loop and not self._loop.is_closed():
            self._shutdown_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)


class SearchEngine:
    """Simple search engine wrapper"""
    
    def __init__(self):
        self.engine: Optional[PardoEngine] = None
        self.settings = get_settings_from_project_root()
        self.resource = Resource(self.settings)
        
    def initialize(self, config: str = "pardo"):
        """Initialize search engine with config"""
        if not self.engine:
            self.engine = PardoEngine.new(self.settings, self.resource, config)
            
    async def search(self, query: str, lang: str = "English") -> AsyncGenerator:
        """Search and yield results"""
        if not self.engine:
            self.initialize()
            
        event_deps = TaskEventDeps()
        async for event in event_deps.consume(
            lambda: self.engine.summary_qa(event_deps, query, lang)
        ):
            yield event


def run_async_search(engine: SearchEngine, query: str, lang: str) -> queue.Queue:
    """Run async search in background thread"""
    result_queue = queue.Queue()
    
    async def search_task():
        try:
            async for event in engine.search(query, lang):
                result_queue.put(("data", event))
        except Exception as e:
            result_queue.put(("error", e))
        finally:
            result_queue.put(("done", None))
    
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(search_task())
        loop.close()
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    
    return result_queue


def process_search_results(result_queue: queue.Queue, placeholders: dict):
    """Process search results and update UI"""
    final_result = ""
    sources = []
    
    while True:
        try:
            result_type, data = result_queue.get(timeout=1.0)
            
            if result_type == "error":
                st.error(f"Search failed: {data}")
                break
                
            elif result_type == "done":
                break
                
            elif result_type == "data":
                event = data
                
                # Handle final result
                if isinstance(event, EndResult):
                    placeholders["thinking"].empty()
                    with placeholders["result"].container():
                        st.markdown("### Answer")
                        st.markdown(event.res)
                    break
                
                # Handle event data
                event_data, is_result = event
                
                if not is_result:
                    # Show thinking status
                    with placeholders["thinking"].container():
                        st.info("🤔 Processing...")
                        
                elif isinstance(event_data, EventTaskOutput):
                    # Show sources
                    sources = event_data.task_output
                    with placeholders["sources"].container():
                        st.markdown("### Sources")
                        st.success(f"Found {len(sources)} sources")
                        
                        with st.expander("View details"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}**")
                                if hasattr(source, "url"):
                                    st.markdown(f"- {source.url}")
                                if hasattr(source, "title"):
                                    st.markdown(f"- {source.title}")
                                st.divider()
                                
                elif isinstance(event_data, EventTaskOutputStream):
                    # Stream started
                    with placeholders["result"].container():
                        st.markdown("### Answer")
                        st.markdown("*Generating...*")
                        
                elif isinstance(event_data, EventTaskOutputStreamDelta):
                    # Stream content
                    if hasattr(event_data, "get_text"):
                        text = event_data.get_text()
                        if text:
                            final_result += text
                            with placeholders["result"].container():
                                st.markdown("### Answer")
                                st.markdown(final_result)
                                
        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error: {e}")
            break


def main():
    st.set_page_config(
        page_title="Simplicity Search",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 Simplicity")
    st.markdown("Simple web search engine")
    
    # Initialize session state
    if "engine" not in st.session_state:
        st.session_state.engine = SearchEngine()
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        # Language selection
        languages = ["auto", None, "English", "中文", "日本語", "Français", "Deutsch", "Español"]
        search_lang = st.selectbox("Search Language", languages)
        
        # Engine config
        try:
            settings = get_settings_from_project_root()
            configs = list(settings.engine_configs.keys())
        except:
            configs = ["pardo"]
            
        engine_config = st.selectbox("Engine Config", configs)
        
        # Initialize engine with selected config
        st.session_state.engine.initialize(engine_config)
    
    # Search interface
    query = st.text_input(
        "Search query:",
        placeholder="What would you like to know?"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if query.strip():
            # Create placeholders
            placeholders = {
                "thinking": st.empty(),
                "sources": st.empty(),
                "result": st.empty()
            }
            
            # Run search
            result_queue = run_async_search(
                st.session_state.engine,
                query,
                search_lang
            )
            
            # Process results
            process_search_results(result_queue, placeholders)
    
    # Example queries
    if not query:
        st.markdown("### Example Queries")
        
        col1, col2, col3 = st.columns(3)
        
        examples = [
            ("🤖 AI developments", "What are the latest AI developments?"),
            ("🌍 Climate change", "Current climate change impacts"),
            ("💻 Programming", "Best practices for Python programming")
        ]
        
        for i, (icon_text, example_query) in enumerate(examples):
            col = [col1, col2, col3][i]
            with col:
                if st.button(icon_text, use_container_width=True):
                    st.session_state.query = example_query
                    st.rerun()


if __name__ == "__main__":
    main()
