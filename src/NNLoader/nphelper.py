def to_array(tensor: TensorProto, base_dir: str = "") -> np.ndarray:
    """Converts a tensor def object to a numpy array.
    Args:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it
    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError("Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    storage_np_dtype = helper.tensor_dtype_to_np_dtype(
        helper.tensor_dtype_to_storage_tensor_dtype(tensor_dtype)
    )
    storage_field = helper.tensor_dtype_to_field(tensor_dtype)
    dims = tensor.dims

    if tensor.data_type == TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = list(s.decode("utf-8") for s in utf8_strings)
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    # Load raw data from external tensor if it exists
    if uses_external_data(tensor):
        load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        if sys.byteorder == "big":
            # Convert endian from little to big
            convert_endian(tensor)

        # manually convert bf16 since there's no numpy support
        if tensor_dtype == TensorProto.BFLOAT16:
            data = np.frombuffer(tensor.raw_data, dtype=np.int16)
            return bfloat16_to_float32(data, dims)

        return np.frombuffer(tensor.raw_data, dtype=np_dtype).reshape(dims)  # type: ignore[no-any-return]

    # float16 is stored as int32 (uint16 type); Need view to get the original value
    if tensor_dtype == TensorProto.FLOAT16:
        return (
            np.asarray(tensor.int32_data, dtype=np.uint16)
            .reshape(dims)
            .view(np.float16)
        )

    # bfloat16 is stored as int32 (uint16 type); no numpy support for bf16
    if tensor_dtype == TensorProto.BFLOAT16:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return bfloat16_to_float32(data, dims)

    data = getattr(tensor, storage_field)
    if tensor_dtype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        data = combine_pairs_to_complex(data)  # type: ignore[assignment,arg-type]

    return np.asarray(data, dtype=storage_np_dtype).astype(np_dtype).reshape(dims