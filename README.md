## Audio Loader

![audio_loader](https://github.com/danielgural/semantic_video_search/blob/main/assets/audio.gif)

This FiftyOne plugin is a Python plugin that allows for you to load in your audio datasets as spectograms in FiftyOne!

🎧 Explore a whole new modality with FiftyOne!

## Installation

```shell
fiftyone plugins download https://github.com/danielgural/audio_loader
```

## Operators

### `load_audio`

Loads audio files in that are saved as ".wav" files based on classification directory tree. Such as:

```
<dataset_dir>/
    <classA>/
        <audio1>.wav
        <audio2>.wav
        ...
    <classB>/
        <audio1>.wav
        <audio2>.wav
        ...
    ...
```

When loading, spectograms will be created for each audio file and saved to the output directory. For example, if input directory = output directory:

```
<dataset_dir>/
    <classA>/
        <audio1>.wav
        <audio2>.wav
        ...
    <classsA_spectograms>/
        <spectogram1>.png
        <spectogram2>.png
        ...
    <classB>/
        <audio1>.wav
        <audio2>.wav
        ...
    <classsB_spectograms>/
        <spectogram1>.png
        <spectogram2>.png
        ...
    ...
```

Additionally, frame_rate, duration, channels, sample_width, and total_frame_count will be added as fields to the sample as well! 



