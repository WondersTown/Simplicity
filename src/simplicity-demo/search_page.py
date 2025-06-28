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
            future = asyncio.run_coroutine_threadsafe(collect_results(), loop)
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
    """Wrapper class to manage the search engine instance"""

    def __init__(self, engine_config: str = "pardo"):
        self.engine: Optional[PardoEngine] = None
        self.initialized = False
        self.engine_config = engine_config
        self.settings = None

    async def initialize(self):
        """Initialize the search engine"""
        if not self.initialized:
            try:
                self.settings = get_settings_from_project_root()
                resource = Resource(self.settings)
                self.engine = PardoEngine.new(self.settings, resource, self.engine_config)
                self.initialized = True
                return True
            except Exception as e:
                # Note: Can't use st.error here as we're in async context
                raise RuntimeError(
                    f"Failed to initialize search engine with config '{self.engine_config}': {e!s}"
                ) from None
        return True

    async def reinitialize_with_config(self, engine_config: str):
        """Reinitialize the search engine with a new config"""
        if self.engine_config != engine_config:
            self.engine_config = engine_config
            self.initialized = False
            self.engine = None
            await self.initialize()

    async def search(self, query: str, search_lang: str = "English") -> AsyncGenerator:
        """Perform search and yield results"""
        if not self.initialized:
            await self.initialize()

        if self.engine is None:
            raise RuntimeError("Search engine not initialized")

        event_deps = TaskEventDeps()
        async for event in event_deps.consume(
            lambda: self.engine.summary_qa(event_deps, query, search_lang)  # type: ignore
        ):
            yield event


# Initialize the global async loop manager in session state
def get_loop_manager():
    """Get or create the async loop manager from session state"""
    if "async_loop_manager" not in st.session_state:
        st.session_state.async_loop_manager = AsyncLoopManager()
        # Register cleanup on app exit
        atexit.register(st.session_state.async_loop_manager.shutdown)
    return st.session_state.async_loop_manager


def run_search_with_ui_updates(
    search_engine,
    query,
    search_language,
    thinking_placeholder,
    sources_placeholder,
    result_placeholder,
):
    """Run search with UI updates in the main thread"""
    loop_manager = get_loop_manager()
    result_queue = queue.Queue()

    # Start the async search in the background
    async def search_generator():
        return search_engine.search(query, search_language)

    # Run the async generator in the background thread
    loop_manager.run_async_generator(
        lambda: search_engine.search(query, search_language), result_queue
    )

    # Process results in the main thread (with Streamlit context)
    thinking_count = 0
    sources_found = False
    final_result = ""
    start_time = time.time()
    max_timeout = 300  # 5 minutes maximum timeout

    try:
        while True:
            # Check for overall timeout
            if time.time() - start_time > max_timeout:
                raise TimeoutError("Search operation timed out after 5 minutes")
            try:
                # Get result with timeout to avoid blocking indefinitely
                result_type, data = result_queue.get(timeout=1.0)

                if result_type == "error":
                    raise data
                elif result_type == "done":
                    break
                elif result_type == "data":
                    event = data

                    if isinstance(event, EndResult):
                        # Final result
                        final_result = event.res
                        thinking_placeholder.empty()

                        with result_placeholder.container():
                            st.markdown("### 📝 Answer")
                            with st.container():
                                st.markdown(final_result)
                        break

                    event_data, is_result = event

                    if not is_result:
                        # Thinking indicator
                        thinking_count += 1
                        with thinking_placeholder.container():
                            st.info(f"🤔 Processing step {thinking_count}... (This may take a moment)")
                            st.caption("The engine is analyzing sources and preparing your answer.")

                    elif isinstance(event_data, EventTaskOutput):
                        # Sources found
                        if not sources_found:
                            sources_found = True
                            with sources_placeholder.container():
                                st.markdown("### 📚 Sources Found")
                                if hasattr(event_data.task_output, "__len__"):
                                    st.success(
                                        f"Found {len(event_data.task_output)} relevant sources"
                                    )

                                    # Show source details in an expander
                                    with st.expander(
                                        "View Source Details", expanded=False
                                    ):
                                        for i, source in enumerate(
                                            event_data.task_output
                                        ):
                                            with st.container():
                                                st.markdown(f"**Source {i + 1}:**")
                                                if hasattr(source, "url"):
                                                    st.markdown(f"- URL: {source.url}")
                                                if hasattr(source, "title"):
                                                    st.markdown(
                                                        f"- Title: {source.title}"
                                                    )
                                                if (
                                                    hasattr(source, "publishedTime")
                                                    and source.publishedTime
                                                ):
                                                    st.markdown(
                                                        f"- Published: {source.publishedTime}"
                                                    )
                                                if i < len(event_data.task_output) - 1:
                                                    st.divider()

                    elif isinstance(event_data, EventTaskOutputStream):
                        # Stream started
                        with result_placeholder.container():
                            st.markdown("### 📝 Answer")
                            st.markdown("*Generating response...*")

                    elif isinstance(event_data, EventTaskOutputStreamDelta):
                        # Stream content
                        if hasattr(event_data, "get_text"):
                            text = event_data.get_text()
                            if text:
                                final_result += text
                                with result_placeholder.container():
                                    st.markdown("### 📝 Answer")
                                    with st.container():
                                        st.markdown(final_result)

            except queue.Empty:
                # No new results, continue waiting
                # Update the thinking indicator to show we're still processing
                if thinking_count > 0:
                    with thinking_placeholder.container():
                        st.info(f"🤔 Processing step {thinking_count}... (This may take a moment)")
                        st.caption("The engine is analyzing sources and preparing your answer.")
                time.sleep(0.5)
                continue

    except Exception as e:
        thinking_placeholder.empty()
        st.error(f"Search failed: {e!s}")
        st.exception(e)


# Initialize session state
if "search_engine" not in st.session_state:
    st.session_state.search_engine = SearchEngine()

if "current_search" not in st.session_state:
    st.session_state.current_search = None

if "search_language" not in st.session_state:
    st.session_state.search_language = "English"

if "engine_config" not in st.session_state:
    st.session_state.engine_config = "pardo"


def main():
    st.set_page_config(
        page_title="Simplicity - Free and Open-Source Web Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🔍 Simplicity")
    st.markdown("#### Make everything simple instead of perplexing.")

    # Sidebar
    with st.sidebar:
        st.header("Search Settings")

        # Language options with native names only
        language_options = [
            "Same as query",
            "Other...",
            "English",
            "中文",
            "日本語",
            "Français",
            "Deutsch",
            "Español",
            "Русский",
            "العربية",
            "한국어",
            "Português",
            "Italiano",
            "Nederlands",
        ]

        # Simple selectbox for language selection
        try:
            if st.session_state.search_language in language_options:
                current_index = language_options.index(st.session_state.search_language)
            else:
                current_index = (
                    2  # Default to "English" if current language not in predefined list
                )
        except ValueError:
            current_index = 2  # Default to English if there's an error

        selected_language_option = st.selectbox(
            "Search Language",
            options=language_options,
            index=current_index,
            help="Language for web search queries",
        )

        # Handle "None" and "Other..." selections
        if selected_language_option == "Same as query":
            search_language = None
        elif selected_language_option == "Other...":
            search_language = st.text_input(
                "Enter custom language:",
                value=st.session_state.search_language
                if st.session_state.search_language not in language_options[2:]
                else "",
                placeholder="e.g., हिन्दी, Türkçe, Svenska...",
                help="Type any language name",
            )
        else:
            search_language = selected_language_option

        # Update session state when language changes
        if search_language and search_language != st.session_state.search_language:
            st.session_state.search_language = search_language

        # Engine Configuration Selection
        st.markdown("---")  # Add a separator
        
        # Load available engine configs
        try:
            settings = get_settings_from_project_root()
            available_engine_configs = list(settings.engine_configs.keys())
        except Exception as e:
            st.error(f"Failed to load engine configurations: {e}")
            available_engine_configs = ["pardo"]  # Fallback
        
        if not available_engine_configs:
            available_engine_configs = ["pardo"]  # Fallback if empty
            
        # Find current index
        try:
            current_config_index = available_engine_configs.index(st.session_state.engine_config)
        except ValueError:
            current_config_index = 0  # Default to first available
            st.session_state.engine_config = available_engine_configs[0]

        selected_engine_config = st.selectbox(
            "Engine Configuration",
            options=available_engine_configs,
            index=current_config_index,
            help="Select the engine configuration to use for search processing",
        )

        # Update session state and reinitialize engine if config changes
        if selected_engine_config != st.session_state.engine_config:
            st.session_state.engine_config = selected_engine_config
            # Reset the search engine to force reinitialization with new config
            st.session_state.search_engine.engine_config = selected_engine_config
            st.session_state.search_engine.initialized = False
            st.session_state.search_engine.engine = None
            st.success(f"Switched to engine configuration: {selected_engine_config}")

    # Main search interface
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.current_search or "",
        placeholder="What would you like to know?",
        key="search_input",
    )

    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Handle search
    if search_button and query.strip():
        st.session_state.current_search = query

        # Perform search
        with st.container():
            st.markdown(f"### 🔍 Searching for: *{query}*")

            # Create placeholders for different types of content
            thinking_placeholder = st.empty()
            sources_placeholder = st.empty()
            result_placeholder = st.empty()

            # Run the search with UI updates in the main thread
            try:
                run_search_with_ui_updates(
                    st.session_state.search_engine,
                    query,
                    search_language,
                    thinking_placeholder,
                    sources_placeholder,
                    result_placeholder,
                )
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"Search failed: {e!s}")
                st.error("This might be due to:")
                st.markdown("""
                - Network connectivity issues
                - API rate limits or authentication problems
                - Engine configuration issues
                - Temporary service unavailability
                
                Please try again in a few moments or check your configuration.
                """)
                if st.button("🔄 Retry Search"):
                    st.rerun()

    # Examples section
    if not st.session_state.current_search:
        st.markdown("### 💡 Example Queries")

        # English examples (multilingual queries with English search)
        st.markdown(
            "Using your native language to query and answer, but search in another language"
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "🗾 (日) Working visa detailed requirements in Japan",
                use_container_width=True,
            ):
                st.session_state.current_search = (
                    "Working visa detailed requirements in Japan"
                )
                st.session_state.search_language = "日本語"
                st.rerun()

        with col2:
            if st.button("🤖 (En) 人工智能最新发展？", use_container_width=True):
                st.session_state.current_search = "人工智能最新发展？"
                st.session_state.search_language = "English"
                st.rerun()

        with col3:
            if st.button(
                "🌍 (中) How to take subway in Shanghai", use_container_width=True
            ):
                st.session_state.current_search = "How to take subway in Shanghai"
                st.session_state.search_language = "中文"
                st.rerun()

        # Multi-language examples
        st.markdown("Using your native language to query, search and answer")
        col4, col5, col6 = st.columns(3)

        with col4:
            if st.button("🗾 日本の首都は？", use_container_width=True):
                st.session_state.current_search = "日本の首都は？"
                st.session_state.search_language = "Same as query"
                st.rerun()

        with col5:
            if st.button("🤖 人工智能最新发展？", use_container_width=True):
                st.session_state.current_search = "人工智能最新发展？"
                st.session_state.search_language = "Same as query"
                st.rerun()

        with col6:
            if st.button("🌍 Événements mondiaux actuels", use_container_width=True):
                st.session_state.current_search = "Événements mondiaux actuels"
                st.session_state.search_language = "Same as query"
                st.rerun()


if __name__ == "__main__":
    main()
