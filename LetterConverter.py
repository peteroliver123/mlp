class LetterConverter:
    @staticmethod
    def convert_letter(letter):
        numerical_value = ord(letter)
        array_index_one = numerical_value - 65
        alphabet = [0 for i in range(0, 26)]
        alphabet[array_index_one] = 1
        return alphabet
