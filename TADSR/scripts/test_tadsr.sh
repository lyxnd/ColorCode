python test_tadsr.py \
--input_image /mnt/d/leiyuechuan_data/HighlightRetinex/data/sid_highlight/test/low/ \
--output_dir ./output/sid/t102/skip1 \
--tadsr_path ./experience/toLLIE_colorpath/checkpoints/model_best.pkl \
--pretrained_model_name_or_path ./weights \
--ram_ft_path ./weights/DAPE.pth \
--ram_path ./weights/ram_swin_large_14m.pth \
--upscale 1 \
--align_method nofix \
--process_size 0 \
--ram_brighten 0.4 \
--timesteps 102 \
--skip_scale 1 \
--latent_mode mean

#/mnt/d/LOLv2/Real_captured/Test/Low