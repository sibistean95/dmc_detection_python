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

    def correct_errors(self, total_cw, codewords: bytes) -> str:
        if total_cw not in self.ec_table:
            print(f"Unknown length ({total_cw} codeworks)")
            return ""

        ec_codewords = self.ec_table[total_cw]

        print(f"Total codewords: {total_cw}, byte string: {codewords}")

        rs = reedsolo.RSCodec(ec_codewords)

        try:
            # data_bytes = bytearray(codewords)
            decoded_msg = rs.decode(codewords)[0]
            return decoded_msg.decode("utf-8")
        except reedsolo.ReedSolomonError:
            print("Too many errors")
            return ""

    @staticmethod
    def decode_ascii_scheme(data_bytes: list) -> tuple[str, int]:
        decoded_text = ""
        idx = -1
        for idx, byte in enumerate(data_bytes):
            if byte == 129:
                break
            elif 1 <= byte <= 128:
                decoded_text += chr(byte - 1)
            elif 130 <= byte <= 229:
                pair_val = byte - 130
                decoded_text += str(pair_val).zfill(2)


        return decoded_text, idx

    @staticmethod
    def decode_error_correction_bytes(error_bytes: list) -> list:
        result = []
        for code in error_bytes:
            if code < 242:
                result.append(0xe6 + (code - 230))
        return result


    def decode(self, codewords: list) -> str:
        # corrected_bytes = self.correct_errors(codewords)
        # if not corrected_bytes:
        #     return "DECODING FAILED"

        ascii_string, idx =  self.decode_ascii_scheme(codewords)
        error_codewords = codewords[idx + 1:]
        error_hex = self.decode_error_correction_bytes(error_codewords)
        reed_solomon_input = b''
        reed_solomon_input += ascii_string.encode("utf-8")

        for err_hex in error_hex:
            reed_solomon_input += err_hex.to_bytes()

        result = self.correct_errors(len(codewords), reed_solomon_input)

        return result
