from models.atomaudio import audio_model_runner
from models.atomvision import vision_model_runner
from models.atomsudoku import sudoku_model_runner
from models.atomgpt import text_model_runner

def main():
    # Training Configs
    BATCH_SIZE = 32
    MAX_EPOCHS = 1000
    
    # Models to Run
    # audio_model_runner(MAX_EPOCHS, BATCH_SIZE)
    # vision_model_runner(MAX_EPOCHS, BATCH_SIZE)
    sudoku_model_runner(MAX_EPOCHS, BATCH_SIZE)
    # text_model_runner(MAX_EPOCHS, BATCH_SIZE)

main()
