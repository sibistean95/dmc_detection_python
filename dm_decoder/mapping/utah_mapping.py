import numpy as np
from typing import List

class UtahMapper:
    def __init__(self):
        self.chunk_counter = 0

    @staticmethod
    def read_module(matrix: np.ndarray, nrow: int, ncol: int, row: int, col: int, visited: np.ndarray, bit_index: int, byte_val: int) -> int:
        if row < 0:
            row += nrow
            col += 4 - ((nrow + 4) % 8)
        if col < 0:
            col += ncol
            row += 4 - ((ncol + 4) % 8)

        # print(f"row = {row}, col = {col}")

        visited[row, col] = True

        if matrix[row, col] == 1:
            byte_val |= (1 << (7 - bit_index))

        return byte_val

    def utah(self, matrix: np.ndarray, nrow: int, ncol: int, row: int, col: int, visited: np.ndarray) -> int:
        byte_val = 0
        byte_val = self.read_module(matrix, nrow, ncol, row - 2, col - 2, visited, 0, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row - 2, col - 1, visited, 1, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row - 1, col - 2, visited, 2, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row - 1, col - 1, visited, 3, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row - 1, col, visited, 4, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row, col - 2, visited, 5, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row, col - 1, visited, 6, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, row, col, visited, 7, byte_val)

        self.chunk_counter += 1

        b0 = (byte_val >> 7) & 1
        b1 = (byte_val >> 6) & 1
        b2 = (byte_val >> 5) & 1
        b3 = (byte_val >> 4) & 1
        b4 = (byte_val >> 3) & 1
        b5 = (byte_val >> 2) & 1
        b6 = (byte_val >> 1) & 1
        b7 = byte_val & 1

        print(f"--- Chunk {self.chunk_counter} (row:{row}, col:{col}) ---")
        print(f"[{b0}] [{b1}]")
        print(f"[{b2}] [{b3}] [{b4}]")
        print(f"[{b5}] [{b6}] [{b7}]")
        print(f"Value: {byte_val}\n")

        return byte_val

    # corner special cases
    def corner1(self, matrix: np.ndarray, nrow: int, ncol: int, visited: np.ndarray) -> int:
        byte_val = 0
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 0, visited, 0, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 1, visited, 1, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 2, visited, 2, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 2, visited, 3, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 1, visited, 4, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 1, visited, 5, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 2, ncol - 1, visited, 6, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 3, ncol - 1, visited, 7, byte_val)

        self.chunk_counter += 1
        print(f"--- Chunk {self.chunk_counter} (Corner 1) --- Value: {byte_val}\n")

        return byte_val

    def corner2(self, matrix: np.ndarray, nrow: int, ncol: int, visited: np.ndarray) -> int:
        byte_val = 0
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 3, 0, visited, 0, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 2, 0, visited, 1, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 0, visited, 2, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 4, visited, 3, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 3, visited, 4, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 2, visited, 5, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 1, visited, 6, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 1, visited, 7, byte_val)

        self.chunk_counter += 1
        print(f"--- Chunk {self.chunk_counter} (Corner 2) --- Value: {byte_val}\n")

        return byte_val

    def corner3(self, matrix: np.ndarray, nrow: int, ncol: int, visited: np.ndarray) -> int:
        byte_val = 0
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 3, 0, visited, 0, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 2, 0, visited, 1, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 0, visited, 2, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 2, visited, 3, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 1, visited, 4, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 1, visited, 5, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 2, ncol - 1, visited, 6, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 3, ncol - 1, visited, 7, byte_val)

        self.chunk_counter += 1
        print(f"--- Chunk {self.chunk_counter} (Corner 3) --- Value: {byte_val}\n")

        return byte_val

    def corner4(self, matrix: np.ndarray, nrow: int, ncol: int, visited: np.ndarray) -> int:
        byte_val = 0
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, 0, visited, 0, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, nrow - 1, ncol - 1, visited, 1, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 3, visited, 2, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 2, visited, 3, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 0, ncol - 1, visited, 4, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 3, visited, 5, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 2, visited, 6, byte_val)
        byte_val = self.read_module(matrix, nrow, ncol, 1, ncol - 1, visited, 7, byte_val)

        self.chunk_counter += 1
        print(f"--- Chunk {self.chunk_counter} (Corner 4) --- Value: {byte_val}\n")

        return byte_val

    def map_to_codewords(self, data_matrix: np.ndarray) -> List[int]:
        nrow, ncol = data_matrix.shape
        visited = np.zeros((nrow, ncol), dtype=bool)
        codewords = []

        row = 4
        col = 0

        while True:
            if (row == nrow) and (col == 0):
                codewords.append(self.corner1(data_matrix, nrow, ncol, visited))
            elif (row == nrow - 2) and (col == 0) and (ncol % 4 != 0):
                codewords.append(self.corner2(data_matrix, nrow, ncol, visited))
            elif (row == nrow - 2) and (col == 0) and (ncol % 8 == 4):
                codewords.append(self.corner3(data_matrix, nrow, ncol, visited))
            elif (row == nrow + 4) and (col == 2) and (ncol % 8 == 0):
                codewords.append(self.corner4(data_matrix, nrow, ncol, visited))

            while True:
                if (row < nrow) and (col >= 0) and not visited[row, col]:
                    codewords.append(self.utah(data_matrix, nrow, ncol, row, col, visited))
                row -= 2
                col += 2
                if not ((row >= 0) and (col < ncol)):
                    break

            row += 1
            col += 3

            while True:
                if (row >= 0) and (col < ncol) and not visited[row, col]:
                    codewords.append(self.utah(data_matrix, nrow, ncol, row, col, visited))
                row += 2
                col -= 2
                if not ((row < nrow) and (col >= 0)):
                    break

            row += 3
            col += 1

            if not ((row < nrow) or (col < ncol)):
                break

        if not visited[nrow - 1, ncol - 1]:
            visited[nrow - 1, ncol - 1] = True
        if not visited[nrow - 2, ncol - 2]:
            visited[nrow - 2, ncol - 2] = True

        return codewords