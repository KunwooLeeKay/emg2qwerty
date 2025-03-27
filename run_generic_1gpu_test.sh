python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=1 \
  +trainer.enable_progress_bar=true \
  --multirun