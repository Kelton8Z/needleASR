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