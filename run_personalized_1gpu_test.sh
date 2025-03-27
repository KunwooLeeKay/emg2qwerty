python -m emg2qwerty.train \
   user="glob(user0)" \
   trainer.accelerator=gpu trainer.devices=1 \
   +trainer.enable_progress_bar=true \
   checkpoint="/ocean/projects/cis250053p/clee18/emg2qwerty/models/generic.ckpt" \
   --multirun