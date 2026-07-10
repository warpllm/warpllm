import warpllm


def test_version():
    assert warpllm.version() == warpllm.__version__
