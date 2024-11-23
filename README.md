# needleASR

## Development Notes:
- NDArray does not support 
    ```
        # A: NDArray
        for a in A:
    ```
    only support
    ```
        # A: NDArray
        for i in range(len(A))
    ```

- When creating new NDArray, the shape must be a tuple, could not be an int, 
  even if creating a 1D vector, e.g.
  ```
    extended_symbols = array_api.full((2 * len(target) + 1, ), self.blank, device=target.device) # accepable, (2 * len(target) + 1) false, must end with a ',', 
    (2 * len(target) + 1, ) true
  ```

- NDArray __getitem__ could only create an NDArray with the same shape as before, 
  e.g. A shape (3, 4), A[1, :3] is with shape (1, 3) rather than (3, )

- If want to reshape a getitem sliced NDArray, must first compact() than reshape
  target_trunc = target[batch_itr, :int(target_lengths[batch_itr].numpy())].compact().reshape(-1)

- Only support compact with positive offset

- a: NDArray([1.0]) != 1.0, cannot directly compare NDArray with number, must use float(a.numpy()) != 1.0