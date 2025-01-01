from functools import wraps
from typing import Callable, Optional, TypeVar, Union, overload

T = TypeVar("T")


@overload
def update_fn(
    name_or_func: Union[str, Callable[..., T]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


@overload
def update_fn(
    func: Callable[..., T],
    *,
    name: str,
    description: Optional[str] = None,
) -> Callable[..., T]: ...


def update_fn(
    name_or_func: Union[str, Callable[..., T], None] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[Callable[[Callable[..., T]], Callable[..., T]], Callable[..., T]]:
    """Rename a function and optionally set its docstring.

    Can be used as a decorator or called directly on a function.

    Args:
        name_or_func: Either the new name (when used as decorator) or the function to rename
        name: The new name (when used as a function)
        description: Optional docstring for the function

    Example:
        # As decorator with positional arg:
        @update_fn('hello_there', description='Says hello')
        def my_fn(x):
            return x

        # As decorator with keyword args:
        @update_fn(name='hello_there', description='Says hello')
        def my_fn(x):
            return x

        # As function:
        def add_stuff(x):
            return x + 1
        new_fn = update_fn(add_stuff, name='add_stuff_123', description='Adds stuff')
    """

    def apply(func: Callable[..., T], new_name: str) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return func(*args, **kwargs)

        wrapper.__name__ = new_name
        if description is not None:
            wrapper.__doc__ = description
        return wrapper

    if callable(name_or_func):
        # Used as function
        if name is None:
            raise ValueError("name must be provided when used as a function")
        return apply(name_or_func, name)
    else:
        # Used as decorator
        decorator_name = name_or_func if name_or_func is not None else name
        if decorator_name is None:
            raise ValueError("name must be provided either as argument or keyword")

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            return apply(func, decorator_name)

        return decorator
