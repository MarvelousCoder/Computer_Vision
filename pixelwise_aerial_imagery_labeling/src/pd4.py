import argparse
from train import train
from utils import load_dataset, load, form_mask
from glcm_sklearn import *
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(
    description="Image Segmentation")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--r1", action="store_true",
                   help="Requisito 1")
group.add_argument("--r2", action="store_true",
                   help="Requisito 2")
parser.add_argument("--imageSize", choices=(100, 200, 250),
                    help="Tamanho da imagem", type=int, default=200)
parser.add_argument("--batchSize", help="Tamanho do batch", type=int,
                    default=4)
parser.add_argument("--freeze", action="store_true",
                    help="Não treinar camadas do encoder",
                    default=False)
parser.add_argument("--randomInit", help="Não usar pesos pretreinados",
                    action="store_true", default=False)
parser.add_argument("--train", help="Treinar modelo",
                    action="store_true", default=False)
parser.add_argument("--model", help="Nome do modelo a ser salvo/carregado",
                    type=str, default="test")


def main(r1, r2, train_model, image_size,
         batch_size, freeze, pretrained, model):
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset(image_size)
    if r1:
        pass
    elif r2:
        if train_model:
            train(X_train, Y_train, X_dev, Y_dev, batch_size=batch_size,
                  freeze=freeze, pretrained=pretrained, model=model)
        deeplab = load(model)
        loss, acc, iou = deeplab.evaluate(X_test, Y_test, batch_size)
        print("Loss:{}, Acc:{}, IoU:{}".format(loss, acc, iou))
        mask = deeplab.predict([X_test], batch_size=batch_size)
        fullMask = form_mask(mask, image_size)
        plt.imshow(fullMask)
        plt.show()
        cv2.imwrite('{}-pred.png'.format(model), fullMask*255)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.train, args.imageSize, args.batchSize,
         args.freeze, not args.randomInit, args.model)
