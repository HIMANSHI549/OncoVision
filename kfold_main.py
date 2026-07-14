from models.kfold_trainer import KFoldTrainer


def main():

    trainer = KFoldTrainer()

    trainer.run()


if __name__ == "__main__":

    main()