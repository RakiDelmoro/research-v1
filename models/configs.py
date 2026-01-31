from dataclasses import dataclass

@dataclass
class VisionConfigs:
    refinement_steps = 3
    neurons_per_mini_pool = 512
    num_mini_pool = 2
    dropout = 0.1
    patch_size = 4
    possible_predictions = 10

@dataclass
class TextConfigs:
    refinement_steps = 6
    neurons_per_mini_pool = 1024
    num_mini_pool = 16
    dropout = 0.1
    context_size = 8
    possible_predictions = 256
    log_result_freq = 10

@dataclass
class AudioConfigs:
    refinement_steps = 3
    num_mini_pool = 8
    neurons_per_mini_pool = 256
    dropout = 0.1
    audio_patch_size = 16
    possible_predictions = 10

@dataclass
class SudokuConfigs:
    refinement_steps = 3
    neurons_per_mini_pool = 512
    num_mini_pool = 8
    dropout = 0.1
    num_chars = 10
    row_positions = 9
    column_positions = 9
    box_positions = 9
    possible_predictions = 9
