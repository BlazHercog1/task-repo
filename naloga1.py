import cv2 as cv
import numpy as np
import time

prev_time = 0

def zmanjsaj_sliko(slika, sirina, visina):
    return cv.resize(slika, (sirina, visina))
    pass

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    rezultat = []
    for y in range(0, slika.shape[0], visina_skatle):
        vrstica=[]
        for x in range(0, slika.shape[1], sirina_skatle):
            podslika= slika[y:y+visina_skatle, x:x+sirina_skatle]
            stevilo_pikslov=prestej_piklse_z_barvo_koze(podslika, barva_koze)
            vrstica.append(stevilo_pikslov)
        rezultat.append(vrstica)
    return rezultat
    pass


def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    maska = cv.inRange(slika, barva_koze[0], barva_koze[1])
    return cv.countNonZero(maska)
    pass


def doloci_barvo_koze(slika,levo_zgoraj,desno_spodaj) -> tuple:
    roi = slika[levo_zgoraj[1]:desno_spodaj[1], levo_zgoraj[0]: desno_spodaj[0]]
    povprecje = np.mean(roi, axis=(0,1))
    odstopanje = np.std(roi, axis=(1,0))
    spodnja_meja=np.maximum(povprecje-odstopanje, 0).astype(int)
    zgornja_meja=np.minimum(povprecje+odstopanje, 255).astype(int)
    return spodnja_meja, zgornja_meja
    pass


if __name__ == '__main__':
    kamera = cv.VideoCapture(0)
    if not kamera.isOpened():
        print('Kamera ni bila odprta.')
    ret, slika = kamera.read()
    if not ret:
        print("Napaka pri zajemu slike!")
        exit()
    cv.namedWindow('Zajem obraza')
    cv.imshow('Zajem obraza', slika)

    rect= cv.selectROI('Zajem obraza', slika, fromCenter=False, showCrosshair=True)
    cv.destroyWindow('Zajem obraza')

    levo_zgoraj = (rect[0], rect[1])
    desno_spodaj = (rect[0]+rect[2], rect[1]+rect[3])
    barva_koze = doloci_barvo_koze(slika, levo_zgoraj, desno_spodaj)  
    while True:
        trenutni_cas = time.time()
        ret, slika = kamera.read()
        if not ret:
            break
        slika = zmanjsaj_sliko(slika, 450, 450)
        skatle = obdelaj_sliko_s_skatlami(slika, 50, 50, barva_koze)
        for i, vrstica in enumerate(skatle):
            for j, vrednost in enumerate(vrstica):
                if vrednost > 40:
                    cv.rectangle(slika, (j*50, i*50), ((j+1)*50, (i+1)*50), (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv.putText(slika, "FPS: {:.2f}".format(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow('Face Detection with FPS', slika)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    kamera.release()
    cv.destroyAllWindows()
    pass