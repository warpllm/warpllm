use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;
use tokio::sync::Mutex;

create_exception!(
    _warpllm,
    WarpLLMNativeError,
    pyo3::exceptions::PyException,
    "Raised by the native layer with a wire-format JSON message; \
     the Python wrapper translates it into typed exceptions."
);

#[pyfunction]
fn version() -> &'static str {
    warpllm::version()
}

async fn run_chat(
    client: Arc<warpllm::JsonClient>,
    request_json: String,
) -> Result<String, String> {
    client
        .chat_completions(&request_json)
        .await
        .map_err(|error| error.to_openai_json())
}

#[pyclass]
struct Client {
    inner: Arc<warpllm::JsonClient>,
}

#[pymethods]
impl Client {
    #[new]
    fn new(config_json: String) -> PyResult<Self> {
        let inner = warpllm::JsonClient::new(&config_json)
            .map_err(|error| WarpLLMNativeError::new_err(error.to_openai_json()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Blocks on the shared tokio runtime with the GIL released — no
    /// `asyncio.run` involved, so this works inside notebooks and scripts
    /// alike and reuses pooled connections across calls.
    fn chat_completions(&self, py: Python<'_>, request_json: String) -> PyResult<String> {
        let client = self.inner.clone();
        py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(run_chat(client, request_json))
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    fn async_chat_completions<'p>(
        &self,
        py: Python<'p>,
        request_json: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            run_chat(client, request_json)
                .await
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    /// Opens a stream, blocking — the sync client's half, mirroring
    /// [`Client::chat_completions`].
    fn chat_completions_stream(
        &self,
        py: Python<'_>,
        request_json: String,
    ) -> PyResult<ChatStream> {
        let client = self.inner.clone();
        py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(open_stream(client, request_json))
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    /// ...and the async client's, which needs its own rather than awaiting the
    /// one above.
    ///
    /// Opening a stream is short but not instant — it is a POST whose headers
    /// the provider may sit on. `block_on` inside a coroutine holds the thread
    /// for that whole time, and the event loop is that thread: every other
    /// task, timer, and callback stops until the response line arrives. The
    /// GIL being released does not help, because it is the LOOP that is
    /// blocked, not the interpreter.
    fn async_chat_completions_stream<'p>(
        &self,
        py: Python<'p>,
        request_json: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            open_stream(client, request_json)
                .await
                .map_err(WarpLLMNativeError::new_err)
        })
    }
}

async fn open_stream(
    client: Arc<warpllm::JsonClient>,
    request_json: String,
) -> Result<ChatStream, String> {
    client
        .chat_completions_stream(&request_json)
        .await
        .map(|inner| ChatStream {
            inner: Arc::new(Mutex::new(inner)),
        })
        .map_err(|error| error.to_openai_json())
}

/// A handle over one streamed reply. The iteration protocol is Python's —
/// `_client.py` wraps this in `__next__` and `__anext__` — so the sync and
/// async clients each get the idiom they already use for whole replies.
#[pyclass]
struct ChatStream {
    /// `Arc<Mutex<_>>` because the async method has to hand the runtime a
    /// `'static` future, which a borrow of `self` is not. The lock also makes
    /// two callers pulling one stream queue rather than race.
    inner: Arc<Mutex<warpllm::JsonChatStream>>,
}

async fn next_chunk(stream: Arc<Mutex<warpllm::JsonChatStream>>) -> Result<Option<String>, String> {
    stream
        .lock()
        .await
        .next()
        .await
        .transpose()
        .map_err(|error| error.to_openai_json())
}

#[pymethods]
impl ChatStream {
    /// The next chunk as JSON, or `None` once the stream ends. Blocks on the
    /// shared runtime with the GIL released, exactly as `chat_completions` does.
    fn next(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let stream = self.inner.clone();
        py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(next_chunk(stream))
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    fn async_next<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let stream = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            next_chunk(stream)
                .await
                .map_err(WarpLLMNativeError::new_err)
        })
    }
}

#[pymodule]
fn _warpllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Client>()?;
    m.add_class::<ChatStream>()?;
    m.add(
        "WarpLLMNativeError",
        m.py().get_type::<WarpLLMNativeError>(),
    )?;
    Ok(())
}
