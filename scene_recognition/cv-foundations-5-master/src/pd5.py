import argparse
from train import train
from evaluate import evaluate,evaluate_bov
from Bag import BOV
from keras.models import load_model
import os

DIRNAME = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description="Image Classification")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--r1", action="store_true",
                   help="Requisito 1")
group.add_argument("--r2", action="store_true",
                   help="Requisito 2")
group.add_argument("--bonus", help="Usar modelo bonus",
                   action="store_true")
parser.add_argument("--hyper", help="Faz testes com hyperparams no BOVW",
                   action="store_true")
parser.add_argument("--batchSize", help="Tamanho do batch", type=int,
                    default=2)
parser.add_argument("--pooling", help="Método de pooling das características",
                    choices=("avg", "max"), default="avg")
parser.add_argument("--numUnits", help="Número de unidades na camada densa",
                    type=int, default=1024)
parser.add_argument("--train", help="Treinar modelo",
                    action="store_true", default=False)
parser.add_argument("--freeze", help="Congelar camadas do extrator",
                    action="store_true", default=False)
parser.add_argument("--model", help="Nome do modelo a ser salvo/carregado",
                    type=str, default="test")
parser.add_argument("--dropProb", help="Probabilidade de dropout",
                    type=float, default=0.)



def main(r1, r2, train_model,
         batch_size, pooling, numUnits, model,
         drop_prob, bonus, freeze, hyper):
    if r1:
        if hyper:
            for h in (150,200,250,300,350):
                print("Testing no_clusters ",h)
                bov = BOV(no_clusters=h)
                bov.trainModel()
                bov.testModel()
                evaluate_bov(bov.labels,bov.predictions,str(h),confusionMatrix=True)
        else:
            bov = BOV(no_clusters=100)
            bov.trainModel()
            bov.testModel()
            evaluate_bov(bov.labels,bov.predictions,"BOVW",confusionMatrix=True)
    elif r2:
        if train_model:
            train(pooling=pooling,
                  num_units=numUnits,
                  batch_size=batch_size,
                  name=model,
                  drop_prob=drop_prob,
                  freeze=freeze)
        clf = load_model(os.path.join(DIRNAME, "models/{}/model.hdf5".format(model)))
        evaluate(clf, batch_size, confusionMatrix=True)
    elif bonus:
        if train_model:
            train(pooling=pooling,
                  num_units=numUnits,
                  batch_size=batch_size,
                  name=model,
                  drop_prob=drop_prob,
                  bonus=True,
                  freeze=freeze)
        clf = load_model(os.path.join(DIRNAME, "models/{}/model.hdf5".format(model)))
        evaluate(clf, batch_size, confusionMatrix=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.train, args.batchSize,
         args.pooling, args.numUnits, args.model, args.dropProb,
         args.bonus, args.freeze,args.hyper)
