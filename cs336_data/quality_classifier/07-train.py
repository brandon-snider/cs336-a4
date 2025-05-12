import fasttext


def main():
    model = fasttext.train_supervised(
        input="data/wiki/quality.train",
        epoch=30,
        lr=0.2,
        # loss="hs",
        # wordNgrams=3,
    )
    model.save_model("out/models/quality.bin")
    print(model.test("data/wiki/quality.valid", k=1))


if __name__ == "__main__":
    main()
