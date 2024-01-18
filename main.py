import os
from shutil import copyfile
import face_recognition

def load_known_face_encodings(source_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
        image_path = os.path.join(source_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

def sort_images_by_person(source_folder, output_folder, known_face_encodings, known_face_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
        image_path = os.path.join(source_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if any(matches):
                first_match_index = matches.index(True)
                person_name = known_face_names[first_match_index]

                person_folder = os.path.join(output_folder, person_name)
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                destination_path = os.path.join(person_folder, filename)
                copyfile(image_path, destination_path)

if __name__ == "__main__":
    source_folder = r"from path"
    output_folder = r"to path"

    known_face_encodings, known_face_names = load_known_face_encodings(source_folder)
    sort_images_by_person(source_folder, output_folder, known_face_encodings, known_face_names)
