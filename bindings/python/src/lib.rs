use pyo3::prelude::*;

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

#[pymodule]
fn _warpllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(echo, m)?)?;
    Ok(())
}
