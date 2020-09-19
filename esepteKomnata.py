

# use CPU insteed GPU
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
from keras.preprocessing import image
from os import walk
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions


class EsepteKomnata():

    def __init__(self):

        # uploaded image to recognize path
        self.image_path = "C:\\Users\\esept\\Downloads\\EsepteKomnataUploads\\"

        # komnata types
        self.komnataTypes = [
            'Ванная',
            'Спальня',
            'Шкаф',
            'Компьютерная',
            'Коридор',
            'Столовая',
            'Лифт',
            'Гараж',
            'Кухня',
            'Прачечная-автомат',
            'Гостиная',
            'Комната для переговоров',
            'Офис',
            'Лестница',
            'Комната ожидания'
        ]

        # loading model and weights
        self.model = load_model('weights\\inception_v2_200px.h5')
        self.model.load_weights('weights\\Weightsinception_v2_200px.h5')


    def getKomnataType(self, imageName):

        print(imageName)

        komnata = self.prepareImage(imageName)

        result = self.model.predict(komnata)

        for i in range (0, len(self.komnataTypes)):

            if result[0][i] >= 0.6:

                #listOfKeys = [key  for (key, value) in dict.items() if value == i]
                #for key  in listOfKeys:
                #    print(key) 
                #    break
                return self.komnataTypes[i]
                

        # # image
        # img = cv2.imread("/home/apollo/Desktop/EsepteCategoria/EsepteCategoria/wwwroot/data/images/" + imageName)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # get text
        # text = pytesseract.image_to_string(img, lang="rus")  #Specify language to look after!

        # # split text         
        # book = [str.split(text, "\n")]

        # l = len(book[0])

        # if l == 0:
        #     book.append("")

        # if l == 1:
        #     book.append("")

        # lh = int(l / 2)

        # firstA = " ".join(book[0][:lh]).title().split()
        # firstB = " ".join(book[0][lh:]).title().split()

        # secondA = " ".join(firstA)
        # secondB = " ".join(firstB)

        # return [secondA, secondB]

    def prepareImage(self, imageName):
        
        komnata = image.load_img(self.image_path + imageName, target_size = (200, 200))
        #komnata = image.load_img("/home/apollo/Downloads/Stroka.kg/" + imageName, target_size = (200, 200))
        #komnata = image.load_img(imageName, target_size = (200, 200))
        komnata = image.img_to_array(komnata)
        komnata = np.expand_dims(komnata, axis = 0)

        return preprocess_input(komnata) # added to check same preds issue


# Testing 

# esko = EsepteKomnata()

# print(esko.getKomnataType('1.jpg'))
# print(esko.getKomnataType('2.jpg'))
# print(esko.getKomnataType('3.jpg'))
# print(esko.getKomnataType('4.jpg'))
# print(esko.getKomnataType('5.jpg'))
# print(esko.getKomnataType('6.jpg'))
# print(esko.getKomnataType('7.jpg'))
# print(esko.getKomnataType('8.jpg'))
# print(esko.getKomnataType('9.jpg'))
# print(esko.getKomnataType('10.jpg'))
# print(esko.getKomnataType('11.jpg'))
# print(esko.getKomnataType('12.jpg'))
