import sys
import logging

def error_message_detail(error, error_detail: sys) -> str:
    """Generate a detailed error message."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by zero error")
        raise CustomException(str(e), sys) from e
