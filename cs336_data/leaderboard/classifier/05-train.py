import fasttext


def main():
    model = fasttext.train_supervised(
        input="/data/c-sniderb/a4-leaderboard/classifier/quality.train",
        epoch=5,
        lr=0.2,
        # loss="hs",
        # wordNgrams=3,
    )
    model.save_model("/data/c-sniderb/a4-leaderboard/classifier/quality.bin")
    print(model.test("/data/c-sniderb/a4-leaderboard/classifier/quality.valid", k=1))


if __name__ == "__main__":
    main()
