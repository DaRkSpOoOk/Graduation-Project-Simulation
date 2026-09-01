# video_io/

Video decode, frame extraction, timestamp handling, FPS normalization, and metadata utilities.

`video_io.reader.inspect_video()` performs a pre-inference decoder pass and
records reported versus actually decoded frame counts, FPS, dimensions,
duration, and timestamp provenance. `iter_video_frames()` prefers FFprobe
`best_effort_timestamp_time` when FFprobe is installed, then OpenCV's
container position timestamp, and only then an FPS/index fallback. It keeps
the source timestamp and the separate strictly increasing millisecond value
used by MediaPipe VIDEO mode.

The reader does not normalize, smooth, interpolate, or overwrite source RGB
frames.
