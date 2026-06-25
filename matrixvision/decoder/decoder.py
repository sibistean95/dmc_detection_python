from typing import Optional

import cv2 as cv
import numpy as np
import reedsolo

from matrixvision.config import DecoderConfig
from matrixvision.data import Decoded, DetectionResult
from matrixvision.debug import DebugSink, NullSink
from matrixvision.decoder.grid_estimation.estimator import GridEstimator
from matrixvision.utils import valid_shape
from matrixvision.viz import draw_module_numbers, draw_module_grid

from matrixvision.decoder.mapping.utah_mapping import UtahMapper


class Decoder:
    def __init__(self, config: DecoderConfig = DecoderConfig(), debug: DebugSink = NullSink()):
        self.config = config
        self.estimator = GridEstimator(margin=self.config.estimator_margin, debug=debug)
        self.debug = debug

        self.ec_table = {
            8: 5,
            12: 7,
            18: 10,
            24: 12,
            32: 14,
            40: 18,
            50: 20,
            60: 24,
            72: 28
        }

        self.C40_BASIC = ["<S1>", "<S2>", "<S3>", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                          "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.TEXT_BASIC = ["<S1>", "<S2>", "<S3>", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                           "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

        self.SHIFT2_CHARS = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<",
                             "=", ">", "?", "@", "[", "\\", "]", "^", "_", "<FNC1>", "<MACRO05>", "<MACRO06>", "<PAD>",
                             "<PAD>"]

        self.SHIFT3_C40 = ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
                           "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "<DEL>"]
        self.SHIFT3_TEXT = ["`", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                            "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "{", "|", "}", "~", "<DEL>"]

        self.X12_BASIC = [
            "\r", "*", ">", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        ]

    def decode(self, image: np.ndarray, detection: DetectionResult) -> Optional[Decoded]:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image should be a numpy array")

        if len(image.shape) != 3:
            raise ValueError("Input image should have 3 channels")

        warp = detection.rectify(image, output_size=self.config.output_size)

        if warp is None:
            return None

        grid_vis = warp.copy()

        gray = cv.medianBlur(cv.cvtColor(warp, cv.COLOR_BGR2GRAY), self.config.smoothing)

        self.debug.show("rectified", gray)
        self.debug.pause()

        h, w = gray.shape[:2]

        grid = self.estimator.estimate_grid(gray, inverted=detection.is_inverted)

        if grid is not None:
            col_centres, row_centres = grid
            bits = self.estimator.sample_matrix(gray, grid[0], grid[1], inverted=detection.is_inverted)
            matrix = bits[1:-1, 1:-1]

            draw_module_grid(grid_vis, col_centres, row_centres)
            grid_vis = draw_module_numbers(grid_vis, col_centres, row_centres)
            self.debug.show("final grid", grid_vis)
            self.debug.pause()
        else:
            pitch, score = self.estimator.estimate_pitch(gray)

            if pitch is not None:
                # width and height on which the estimator ran and found the pitch
                w_eff = w - 2 * self.estimator.margin
                h_eff = h - 2 * self.estimator.margin
                nx = int(round(w_eff / pitch))
                ny = int(round(h_eff / pitch))
                matrix = self.estimator.get_matrix_data(gray, w / nx, h / ny, ny, nx)
            else:
                matrix = None

        if matrix is None or not valid_shape(matrix.shape[0], matrix.shape[1], self.config.valid_sizes):
            return None

        self.debug.pause()

        mapper = UtahMapper()
        codewords = mapper.map_to_codewords(matrix)

        text = self.codewords_to_text(codewords)

        return Decoded(detection=detection, text=text, codewords=codewords, matrix=matrix)

    def correct_errors(self, total_cw, codewords: bytes) -> bytes:
        if total_cw not in self.ec_table:
            print(f"Unknown length ({total_cw} codeworks)")
            return b""

        ec_codewords = self.ec_table[total_cw]

        print(f"Total codewords: {total_cw}, byte string: {codewords}")

        rs = reedsolo.RSCodec(ec_codewords, prim=0x12d, fcr=1, generator=2)

        try:
            decoded_msg = rs.decode(codewords)[0]
            return bytes(decoded_msg)
        except reedsolo.ReedSolomonError:
            print("Too many errors")
            return b""

    def decode_ascii_scheme(self, data_bytes: list) -> str:
        MODE_ASCII, MODE_C40, MODE_TEXT, MODE_X12 = "ASCII", "C40", "TEXT", "X12"

        current_mode = MODE_ASCII
        shift_state = 0
        high_bit_flag = False
        decoded_text = ""
        i = 0

        while i < len(data_bytes):
            byte = data_bytes[i]

            if current_mode == MODE_ASCII:
                if byte == 129:
                    break
                elif byte == 230:
                    current_mode = MODE_C40
                    shift_state = 0
                elif byte == 238:
                    current_mode = MODE_X12
                elif byte == 239:
                    current_mode = MODE_TEXT
                    shift_state = 0
                elif byte == 254:
                    pass
                elif 1 <= byte <= 128:
                    char_val = byte - 1
                    if high_bit_flag:
                        char_val += 128
                        high_bit_flag = False

                    if char_val < 32:
                        if char_val == 29:
                            decoded_text += "<GS>"
                        elif char_val == 30:
                            decoded_text += "<RS>"
                        elif char_val == 4:
                            decoded_text += "<EOT>"
                        else:
                            decoded_text += f"<CTRL_{char_val}>"
                    else:
                        decoded_text += chr(char_val)
                elif 130 <= byte <= 229:
                    decoded_text += str(byte - 130).zfill(2)

                i += 1

            elif current_mode in (MODE_C40, MODE_TEXT, MODE_X12):
                if byte == 254:
                    current_mode = MODE_ASCII
                    shift_state = 0
                    i += 1
                    continue

                if i + 1 >= len(data_bytes):
                    break

                byte1 = data_bytes[i]
                byte2 = data_bytes[i + 1]
                i += 2

                V = (byte1 * 256) + byte2 - 1
                C1 = V // 1600
                remainder = V % 1600
                C2 = remainder // 40
                C3 = remainder % 40

                for c in [C1, C2, C3]:
                    if current_mode == MODE_X12:
                        if c == 0:
                            decoded_text += "\r"
                        elif c == 1:
                            decoded_text += "*"
                        elif c == 2:
                            decoded_text += ">"
                        elif c == 3:
                            decoded_text += " "
                        elif c < len(self.X12_BASIC):
                            decoded_text += self.X12_BASIC[c]
                        continue

                    if shift_state == 0:
                        if c == 0:
                            shift_state = 1
                        elif c == 1:
                            shift_state = 2
                        elif c == 2:
                            shift_state = 3
                        elif c == 3:
                            decoded_text += " "
                        else:
                            char_to_add = self.C40_BASIC[c] if current_mode == MODE_C40 else self.TEXT_BASIC[c]
                            if high_bit_flag:
                                char_to_add = chr(ord(char_to_add) + 128)
                                high_bit_flag = False
                            decoded_text += char_to_add

                    elif shift_state == 1:
                        if c == 29:
                            decoded_text += "<GS>"
                        elif c == 30:
                            decoded_text += "<RS>"
                        elif c == 4:
                            decoded_text += "<EOT>"
                        elif c < 32:
                            decoded_text += f"<CTRL_{c}>"
                        else:
                            decoded_text += chr(c)
                        shift_state = 0

                    elif shift_state == 2:
                        if c == 30:
                            high_bit_flag = True
                        elif c < len(self.SHIFT2_CHARS):
                            decoded_text += self.SHIFT2_CHARS[c]
                        shift_state = 0

                    elif shift_state == 3:
                        if current_mode == MODE_C40:
                            decoded_text += self.SHIFT3_C40[c]
                        else:
                            decoded_text += self.SHIFT3_TEXT[c]
                        shift_state = 0

        return decoded_text

    @staticmethod
    def decode_error_correction_bytes(error_bytes: list) -> list:
        result = []
        for code in error_bytes:
            if code < 242:
                result.append(0xe6 + (code - 230))
        return result

    def codewords_to_text(self, codewords: list) -> str:
        corrected_bytes = self.correct_errors(len(codewords), bytes(codewords))
        if not corrected_bytes:
            return "DECODING FAILED"

        return self.decode_ascii_scheme(list(corrected_bytes))
