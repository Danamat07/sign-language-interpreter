import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera nu a putut fi deschisă")
        return

    print("Camera OK. Apasă Q pentru ieșire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Sign Language Interpreter - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
