import streamlit as st
import requests
from PIL import Image
from io import BytesIO

#Конфигурация страницы
st.set_page_config(
    page_title="Классификатор птиц",
    page_icon="🐦",
    layout="centered"
)

# Указываем адрес, где запущен наш бэкенд
BACKEND_URL = "https://bbagk1p9sr35jde6ebvs.containers.yandexcloud.net/predict"

#Интерфейс приложения
st.title("🐦 Классификатор видов птиц")
st.write("Загрузите изображение птицы, и нейросеть определит ее вид. Изображение должно быть в хорошем качестве, птица должна быть хорошо видна." \
"Имейте ввиду, что обучалась модель на 200 видах, так что не всех она сможет распознать ")

uploaded_file = st.file_uploader(
    "Выберите изображение...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Отображаем загруженное изображение
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    st.write("")

    # При нажатии на кнопку отправляем запрос на бэкенд
    if st.button("Определить вид птицы"):
        with st.spinner('Отправка изображения на сервер...'):
           
            img_bytes = uploaded_file.getvalue()
            
            # Создаем данные для POST-запроса
            files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
            
            try:
                # Отправляем запрос на бэкенд
                response = requests.post(BACKEND_URL, files=files)
                
                # Проверяем успешность запроса
                if response.status_code == 200:
                    result = response.json()
                    bird_species = result.get("bird_species", "Не удалось определить").replace("_", " ")
                    confidence = result.get("confidence", "N/A")
                    
                    st.success(f"**Предполагаемый вид:** {bird_species}")
                    st.info(f"**Уверенность модели:** {confidence}")
                else:
                    # Показываем ошибку, если что-то пошло не так
                    st.error(f"Ошибка от сервера: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к серверу. Убедитесь, что бэкенд запущен по адресу http://127.0.0.1:8000")
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")



