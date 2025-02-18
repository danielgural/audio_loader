import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.brain import Similarity
import time
import requests
import glob
from pprint import pprint
import os
import fiftyone.core.utils as fou
import eta.core.utils as etau


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import wave

np = fou.lazy_import("numpy")
sio = fou.lazy_import("scipy.io")
ss = fou.lazy_import("scipy.signal")
plt = fou.lazy_import("matplotlib.pyplot")


class LoadAudio(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_audio",
            label="Load Audio",
            description="Loads in sound data as spectograms by taking in a directory in the format of ImageClassificationTree but with \".wav\" files",
            icon="/assets/audio.svg",
            dynamic=True,
            execute_as_generator=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
    
        ready = _audio_loader_inputs(ctx,inputs)

        if ready:
            _execution_mode(ctx, inputs)
        

        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        ### Your logic here ###
        _audio_loader(ctx)
    
        return {}
    
def _audio_loader_inputs(ctx, inputs):
    file_explorer = types.FileExplorerView(
        choose_dir=True,
        button_label="Choose a directory...",
    )
    prop = inputs.file(
        "input_path",
        description=f"Choose the directory containing wav files (must be in AudioClassificationTree format).",
        label="Input path",
        required=True,
        view=file_explorer,
    )

    input_path = _parse_path(ctx, "input_path")
    if input_path is None:
        return False

    file_explorer = types.FileExplorerView(
        choose_dir=True,
        button_label="Choose a directory...",
    )
    inputs.file(
        "output_dir",
        label="Output directory",
        description="Choose a directory to write the spectogram images to (can be the same as input directory).",
        required=True,
        view=file_explorer,
    )

    output_dir = _parse_path(ctx, "output_dir")
    if output_dir is None:
        return False

    inputs.str("name", label="Dataset name", required=True)


    return True

def wav_to_spectrogram(wav_file,output_dir):

    
            # Load the audio file
    sample_rate, audio_data = sio.wavfile.read(wav_file)

    # Generate a spectrogram
    frequencies, times, Sxx = ss.spectrogram(audio_data, fs=sample_rate)

    # Create a new directory path by joining the existing directory with the new directory name
    spectograms_path = output_dir + "/" + wav_file.split("/")[-2] + "_spectograms"

    if not os.path.exists(spectograms_path):
        os.makedirs(spectograms_path)

    # Display the spectrogram
    plt.figure()
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))  # Using log scale for better visualization
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    image_path = spectograms_path + "/" + os.path.splitext(wav_file.split("/")[-1])[0] + ".png" # more robust way to get the file name
    plt.savefig(image_path)
    plt.close()
    image_path = os.path.abspath(image_path)
    return  image_path

def audio_metadata(filepath):

    
    with wave.open(filepath, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        total_frame_count = wav_file.getnframes()
        duration = total_frame_count / float(frame_rate)
    wav_file.close()
    return [channels, sample_width, frame_rate, total_frame_count, duration]

        

def _audio_loader(ctx):
    input_path = _parse_path(ctx, "input_path")
    output_dir = _parse_path(ctx, "output_dir")
    name = ctx.params.get("name")



    samples = []
    classes = set()


    for relpath in etau.list_files(input_path, recursive=True):
        
        chunks = relpath.split(os.path.sep, 1)
        if len(chunks) == 1:
            continue

        label = chunks[0]
        if label.startswith("."):
            continue
        if "spectograms" in relpath:
            continue

        classes.add(label)

        path = os.path.join(input_path, relpath)
        samples.append((path, label))

    dataset = fo.Dataset(name)
    fo_samples = []

    for sample in samples:

        path,label = sample

        expected_formats = {'RIFF', 'RIFX'}
        if ".wav" in path:
            with open(path, 'rb') as file:
                # Read the first four bytes (RIFF or RIFX)
                file_format = file.read(4).decode('ascii')
            file.close()

            if file_format not in expected_formats:
                print("Incorrect wav format: " + file_format + " on file " + path)
                print("Skipping ...")
                continue
            else:


                image_path = wav_to_spectrogram(path, output_dir)

                fo_sample = fo.Sample(filepath=image_path)
                fo_sample["ground_truth"] = fo.Classification(label=label)
                fo_sample["wav_path"] = path
                channels, sample_width, frame_rate, total_frame_count, duration = audio_metadata(path)

                fo_sample["channels"] = channels
                fo_sample["sample_width"] = sample_width
                fo_sample["frame_rate"] = frame_rate
                fo_sample["total_frame_count"] = total_frame_count
                fo_sample["duration"] = duration

                fo_samples.append(fo_sample)
        else:
            print("Not a wav file on file " + path)
            print("Skipping ...")

    dataset.add_samples(fo_samples)
    dataset.persistent = True
    ctx.trigger("reload_dataset")
    return



def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )

def _parse_path(ctx, key):
    value = ctx.params.get(key, None)
    return value.get("absolute_path", None) if value else None

def register(plugin):
    plugin.register(LoadAudio)
