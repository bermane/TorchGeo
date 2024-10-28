import timeit
import os


def run_seg(trainer, task, datamodule, checkpoint_file):
    if os.path.isfile(checkpoint_file):
        print('Resuming training from previous checkpoint...')
        trainer.fit(
            model=task,
            datamodule=datamodule,
            ckpt_path=checkpoint_file
        )
    else:
        print('Starting training from scratch...')
        trainer.fit(
            model=task,
            datamodule=datamodule,
        )

# def run_seg(trainer, task, train_dataloader, val_dataloader, checkpoint_file):
#     if os.path.isfile(checkpoint_file):
#         print('Resuming training from previous checkpoint...')
#         trainer.fit(
#             task,
#             train_dataloader,
#             val_dataloader,
#             ckpt_path=checkpoint_file
#         )
#     else:
#         print('Starting training from scratch...')
#         trainer.fit(
#             task,
#             train_dataloader,
#             val_dataloader
#         )
