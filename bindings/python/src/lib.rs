use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;

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

#[pyfunction]
fn echo(py: Python<'_>, msg: String) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        warpllm::echo(&msg)
            .await
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })
}

async fn run_chat(client: Arc<warpllm::Client>, request_json: String) -> Result<String, String> {
    let request: warpllm::CreateChatCompletionRequest = serde_json::from_str(&request_json)
        .map_err(|e| warpllm::Error::InvalidInput(e.to_string()).to_wire_json())?;
    let completion = client
        .chat_completion(request)
        .await
        .map_err(|e| e.to_wire_json())?;
    serde_json::to_string(&completion)
        .map_err(|e| warpllm::Error::InvalidInput(e.to_string()).to_wire_json())
}

/// Runs the OpenAI-compatible gateway. `args` are CLI flags passed verbatim
/// to the shared Rust parser (`--host`, `--port`, `--timeout-secs`; see
/// `--help`), so every language wrapper exposes identical flags without its
/// own parsing. `--help` prints usage and returns; otherwise this blocks
/// (GIL released) until SIGINT/SIGTERM. Unlike the Node binding, tokio must
/// own the signals here: Python delivers Ctrl+C by setting a flag the
/// interpreter checks, which never happens while the main thread is blocked
/// inside native code.
#[pyfunction]
fn serve(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    use warpllm_server::config::{Cli, parse_cli};
    match parse_cli(args.into_iter()) {
        Ok(Cli::Print(text)) => {
            print!("{text}");
            Ok(())
        }
        Ok(Cli::Run(config)) => {
            warpllm_server::init_tracing();
            py.detach(move || {
                pyo3_async_runtimes::tokio::get_runtime()
                    .block_on(warpllm_server::serve(config, shutdown_signal()))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
    }
}

/// Resolves on SIGINT or, on unix, SIGTERM (what `docker stop` sends) —
/// the same shape as the binary's `shutdown_signal`, which stays private
/// to its `main.rs`.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }
}

#[pyclass]
struct Client {
    inner: Arc<warpllm::Client>,
}

#[pymethods]
impl Client {
    #[new]
    fn new(config_json: String) -> PyResult<Self> {
        let config: warpllm::ClientConfig = serde_json::from_str(&config_json).map_err(|e| {
            WarpLLMNativeError::new_err(warpllm::Error::InvalidInput(e.to_string()).to_wire_json())
        })?;
        let inner = warpllm::Client::new(config)
            .map_err(|e| WarpLLMNativeError::new_err(e.to_wire_json()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Blocks on the shared tokio runtime with the GIL released — no
    /// `asyncio.run` involved, so this works inside notebooks and scripts
    /// alike and reuses pooled connections across calls.
    fn chat_completion(&self, py: Python<'_>, request_json: String) -> PyResult<String> {
        let client = self.inner.clone();
        py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(run_chat(client, request_json))
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    fn async_chat_completion<'p>(
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
}

#[pymodule]
fn _warpllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(echo, m)?)?;
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    m.add_class::<Client>()?;
    m.add(
        "WarpLLMNativeError",
        m.py().get_type::<WarpLLMNativeError>(),
    )?;
    Ok(())
}
