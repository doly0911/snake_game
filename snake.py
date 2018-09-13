
#importaciones a utilizar
import cv2
import numpy as np
import random
import math

#inicializando el texto para mostrar en pantalla
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#Se carga la imagen de la manzana y creamos su máscara para superponerla en la alimentación del video
apple = cv2.imread("apple_img.png", cv2.IMREAD_UNCHANGED)    
apple_mask = apple[:,:,3]
apple_mask_inv = cv2.bitwise_not(apple_mask)
apple = apple[:,:,0:3]

# Se cambia el tamaño de la img de la manzana
apple = cv2.resize(apple,(40,40),interpolation=cv2.INTER_AREA)
apple_mask = cv2.resize(apple_mask,(40,40),interpolation=cv2.INTER_AREA)
apple_mask_inv = cv2.resize(apple_mask_inv,(40,40),interpolation=cv2.INTER_AREA)

#capturando video desde la webcam
video = cv2.VideoCapture(0)

#Kernel para las transformaciones morfologicas
kernel = np.ones((5,5), np.uint8)
kernelClose = np.ones((20,20), np.uint8)

#detectamos el color azul aguamarina
def detect_blue(hsv):
    lower = np.array([90,158,124])
    upper = np.array([110,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_open= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernelClose)
    maskBlue= mask_close;
    return maskBlue

#funciones para detectar la intersección de segmentos de línea.
# p, q, r, s son los parametros de la cabeza y el cuerpo.
def orientation(p,q,r):
    val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        #lineal
        return 0
    elif (val>0):
        #sentido horario
        return 1
    else:
        #anti-horario
        return 2

def intersect(p,q,r,s):
    o1 = orientation(p, q, r)
    o2 = orientation(p, q, s)
    o3 = orientation(r, s, p)
    o4 = orientation(r, s, q)
    if(o1 != o2 and o3 != o4):
        return True

    return False

#q usado para la inicialización de puntos, snake_len para tamaño de la serpiente,
#score para puntauación del juego y temp variable de control para fin del juego    
q,snake_len,score,temp=0,200,0,1

#almacena el punto central de la burbuja roja
point_x,point_y = 0,0

#almacena los puntos que satisfacen la condición, dist almacena la distancia entre 2 pts consecutivos, 
#la longitud en que crecerá la serpiente
last_point_x,last_point_y,dist,length = 0,0,0,0

#almacena todos los puntos del cuerpo de la serpiente.
points = []

#almacena la longitud entre todos los puntos
list_len = []

#se generan números randoms para posicionar la manzana.
random_x = random.randint(10,550)
random_y = random.randint(10,400)

#variables usadas para chequear intercepciones
a,b,c,d = [],[],[],[]

#Ciclo principal
while 1:
    #Se inicializan las siguientes variables para almacenar coordenadas del rectangulo    
    xr, yr, wr, hr = 0, 0, 0, 0  
    _,frame = video.read() 
    #Se invierte el frame
    frame = cv2.flip(frame,1)

    #inicializando los puntos aceptados para que no estén en la esquina superior izquierda
    if(q==0 and point_x!=0 and point_y!=0):
        last_point_x = point_x
        last_point_y = point_y
        q=1

    #Convertimos la imagen de RGB a HSV 
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maskBlue = detect_blue(hsv) 
    #Encontramos el contorno del objeto
    _, contour_red, _ = cv2.findContours(maskBlue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Se dibuja un rectángulo alrededor de la burbuja o punto aceptado
    try:
        for i in range (0,10):
            #(xr, yr) coordenadas superior izquierda del rectángulo y (w, h) el ancho y alto
            xr, yr, wr, hr = cv2.boundingRect(contour_red[i]) 
            if (wr*hr)>2000:
                break
    except:
        pass
    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)

    #Empezamos con la construccion de la culebrita, hallando el centroide (x,y) del rectangulo
    point_x = int(xr+(wr/2))
    point_y = int(yr+(hr/2))
    # Calculamos la distancia entre el último punto (last point) y el punto actual (point)
    dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
    if (point_x!=0 and point_y!=0 and dist>5):
        #si el punto es aceptado, se agrega a la lista de puntos y se agrega su longitud a list_len
        list_len.append(dist)
        length += dist
        last_point_x = point_x
        last_point_y = point_y
        points.append([point_x, point_y])
    #si la longitud es mayor que la longitud esperada, se eliminan puntos de la parte posterior para disminuir la longitud
    if (length>=snake_len):
        for i in range(len(list_len)):
            length -= list_len[0]
            list_len.pop(0)
            points.pop(0)
            if(length<=snake_len):
                break
    #inicializamos la imagen negra para su posterior uso
    blank_img = np.zeros((480, 640, 3), np.uint8)
    #Se dibuja una línea entre todos los puntos
    for i,j in enumerate(points):
        if (i==0):
            continue
        cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), 5)
    cv2.circle(blank_img, (last_point_x, last_point_y), 5 , (10, 200, 150), -1)
    #si la serpiente come una manzana, aumenta la puntuación y encuentra una nueva posición para la manzana
    if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
        score +=1              #incrementa la puntuacion
        snake_len += 25        #crece 25px la serpiente
        random_x = random.randint(10, 550)
        random_y = random.randint(10, 400)
    #agregamos una imágen en blanco
    frame = cv2.add(frame,blank_img)

    #se agrega la imagen de la nueva manzana al frame
    roi = frame[random_y:random_y+40, random_x:random_x+40]
    img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
    img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
    dst = cv2.add(img_bg, img_fg)
    frame[random_y:random_y + 40, random_x:random_x + 40] = dst
    cv2.putText(frame, str("Score - "+str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #verificamos si la serpiente colisiona consigo misma,
    #se compara la posición de la cabeza con las posiciones existentes del cuerpo.
    if(len(points)>5):
        # a y b son la cabeza de la serpiente, c y d son las variables que recoreran las partes del cuerpo de la serpiente.
        b = points[len(points)-2]
        a = points[len(points)-1]
        for i in range(len(points)-3):
            c = points[i]
            d = points[i+1]
            # si hay intersección entre la cabeza y el cuerpo, finaliza el juego
            if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0): 
                temp = 0       
                break
        if temp==0:
            break
 
    cv2.imshow("frame",frame)
    #esperando que presione ESC el usuario para salir del juego      
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release() #Se libera la cámara.
cv2.destroyAllWindows()  
cv2.putText(frame, str("Game Over!"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(frame, str("Press any key to Exit."), (180, 260), font, 1, (255, 200, 0), 2, cv2.LINE_AA)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()