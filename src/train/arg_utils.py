import ast


def parse_list_argument(raw_value: str | None, *, arg_name: str, element_type: type = str) -> list:
    """Parse list-like CLI values from Python literals or comma-separated strings."""
    if raw_value is None or not raw_value.strip():
        return []

    try:
        parsed = ast.literal_eval(raw_value.strip())
    except (SyntaxError, ValueError):
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]

    if isinstance(parsed, (str, int)):
        parsed = [parsed]
    elif isinstance(parsed, tuple):
        parsed = list(parsed)
    elif not isinstance(parsed, list):
        raise ValueError(
            f"{arg_name} must be a Python list literal, single value, or comma-separated string."
        )

    try:
        if element_type is str:
            result = [str(item).strip() for item in parsed if str(item).strip()]
        elif element_type is int:
            result = [int(item) for item in parsed]
        else:
            result = [element_type(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{arg_name} must contain only {element_type.__name__} values, got: {raw_value!r}"
        ) from exc

    return result


def parse_model_id_list(raw_model_ids: str, *, arg_name: str) -> list[str]:
    model_ids = parse_list_argument(raw_model_ids, arg_name=arg_name, element_type=str)
    if not model_ids:
        raise ValueError(f"At least one model ID must be provided via {arg_name}.")
    return model_ids


__all__ = ["parse_list_argument", "parse_model_id_list"]
