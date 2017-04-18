import pytest
import rfpipe.state

class TestClass_create:

    def test_create0(self):
        state = rfpipe.state.State()
        assert 'version' in state.defined

