_state: dict = {}
 
 
def get_state() -> dict:
    return _state
 
 
def set_state(**kwargs) -> None:
    _state.update(kwargs)
 
 
def clear_state() -> None:
    _state.clear()
 