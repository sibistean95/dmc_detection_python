import reedsolo

class DataMatrixDecoder:
    def __init__(self):
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
        MODE_ASCII, MODE_C40, MODE_TEXT = "ASCII", "C40", "TEXT"

        current_mode = MODE_ASCII
        shift_state = 0
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
                elif byte == 239:
                    current_mode = MODE_TEXT
                    shift_state = 0
                elif byte == 254:
                    pass
                elif 1 <= byte <= 128:
                    char_val = byte - 1
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

            elif current_mode in (MODE_C40, MODE_TEXT):
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
                            decoded_text += self.C40_BASIC[c] if current_mode == MODE_C40 else self.TEXT_BASIC[c]
                    elif shift_state == 1:
                        decoded_text += chr(c)
                        shift_state = 0
                    elif shift_state == 2:
                        if c < len(self.SHIFT2_CHARS): decoded_text += self.SHIFT2_CHARS[c]
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

    def decode(self, codewords: list) -> str:
        corrected_bytes = self.correct_errors(len(codewords), bytes(codewords))
        if not corrected_bytes:
            return "DECODING FAILED"

        return self.decode_ascii_scheme(list(corrected_bytes))
