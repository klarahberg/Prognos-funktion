import flet as ft
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class PassengerPredictor:
    def __init__(self):
        self.model = None
        self.load_data()
        self.train_model()
    
    def load_data(self):
        # Läs in dataset med korrekta filnamn
        self.temp_data = pd.read_csv("Sheet1+ (Flera anslutningar)_Lufttemp.csv")
        self.passenger_data = pd.read_csv("Sheet1+ (Flera anslutningar)_Passagerare.csv")
        
        # Förbehandla temperaturdata
        self.temp_data['date'] = pd.to_datetime(self.temp_data['Datum (Sheet11)']).dt.date
        self.temp_data['temperature'] = self.temp_data['Lufttemperatur']
        
        # Förbehandla passagerardata med tid
        self.passenger_data['datetime'] = pd.to_datetime(self.passenger_data['Tidpunkt'])
        self.passenger_data['hour'] = self.passenger_data['datetime'].dt.hour
        self.passenger_data['date'] = self.passenger_data['datetime'].dt.date
        self.passenger_data['passengers'] = pd.to_numeric(self.passenger_data['Totalt antal påstigande'], errors='coerce')
        
        # Filtrera endast UT-resor för passagerardata
        self.passenger_data = self.passenger_data[self.passenger_data['Väg'] == 'UT']
        
        # Konvertera date-kolumnerna till samma typ (datetime64[ns])
        self.temp_data['date'] = pd.to_datetime(self.temp_data['date'])
        self.passenger_data['date'] = pd.to_datetime(self.passenger_data['date'])
        
        # Aggregera till timvisa värden per dag
        self.hourly_data = pd.merge(
            self.temp_data.groupby('date')['temperature'].mean().reset_index(),
            self.passenger_data.groupby(['date', 'hour'])['passengers'].mean().reset_index(),
            on='date'
        )
        
        # Hantera eventuella saknade värden
        self.hourly_data = self.hourly_data.dropna()
    
    def train_model(self):
        X = self.hourly_data[['temperature', 'hour']]
        y = self.hourly_data['passengers']
        
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X, y)
    
    def predict(self, temperature, hour):
        return min(self.model.predict([[temperature, hour]])[0], 163)  # Max 163 passagerare

def get_weather_forecast(date):
    # Koordinater för Göteborg
    lat, lon = "57.70", "11.97"
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&timezone=Europe/Stockholm"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        forecast_dates = data['daily']['time']
        temperatures = data['daily']['temperature_2m_max']
        
        # Hitta prognosen för valt datum
        try:
            index = forecast_dates.index(date.strftime("%Y-%m-%d"))
            return temperatures[index]
        except ValueError:
            return None
    return None

class BoatApp:
    def __init__(self):
        self.predictor = PassengerPredictor()
        self.selected_date = None
        self.selected_time = None
        
    def main(self, page: ft.Page):
        page.title = "Båt 281 - Prognos"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 600
        page.window_height = 800
        page.padding = 20
        
        # Skapa en container för väderikon
        weather_icon = ft.Icon(
            name=ft.icons.WB_SUNNY_OUTLINED,
            size=50,
            color=ft.colors.ORANGE
        )
        
        # Skapa tidväljare
        time_dropdown = ft.Dropdown(
            label="Välj avgångstid",
            width=200,
            options=[
                ft.dropdown.Option(f"{hour:02d}:00", f"{hour:02d}:00")
                for hour in range(6, 23)  # Avgångar mellan 06:00 och 22:00
            ]
        )
        
        prediction_text = ft.Text(
            size=20,
            text_align=ft.TextAlign.CENTER
        )
        
        date_picker = ft.DatePicker(
            on_change=lambda e: self.handle_date_selected(e, time_dropdown, prediction_text),
            first_date=datetime.now(),
            last_date=datetime.now() + timedelta(days=16)
        )
        
        page.overlay.append(date_picker)
        
        select_date_button = ft.ElevatedButton(
            "Välj datum",
            on_click=lambda _: date_picker.pick_date(),
            icon=ft.icons.CALENDAR_TODAY,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE
            )
        )
        
        # Lägg till en header med information
        header_text = ft.Text(
            "Välkommen till Båt 281 Prognos!\nVälj datum och tid för att se förväntad beläggning.",
            size=16,
            color=ft.colors.GREY_800,
            text_align=ft.TextAlign.CENTER
        )
        
        # Lägg till information om datakällor
        footer_text = ft.Text(
            "Data: Väderprognos från Open-Meteo API\nPassagerardata baserad på historisk statistik",
            size=12,
            color=ft.colors.GREY_600,
            text_align=ft.TextAlign.CENTER
        )
        
        # Uppdatera prediktion när tid väljs
        time_dropdown.on_change = lambda e: self.handle_time_selected(e, prediction_text)
        
        page.add(
            ft.Column(
                controls=[
                    ft.Text(
                        "Båt 281 - Beläggningsprognos",
                        size=30,
                        weight=ft.FontWeight.BOLD,
                        color=ft.colors.BLUE_900,
                        text_align=ft.TextAlign.CENTER
                    ),
                    weather_icon,
                    header_text,
                    select_date_button,
                    time_dropdown,
                    prediction_text,
                    footer_text,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20
            )
        )
    
    def handle_date_selected(self, e, time_dropdown, prediction_text):
        if e.data:
            self.selected_date = datetime.strptime(e.data.split('T')[0], "%Y-%m-%d")
            if self.selected_time:
                self.update_prediction(prediction_text)
    
    def handle_time_selected(self, e, prediction_text):
        if e.data:
            self.selected_time = e.data
            if self.selected_date:
                self.update_prediction(prediction_text)
    
    def update_prediction(self, prediction_text):
        if self.selected_date and self.selected_time:
            selected_hour = int(self.selected_time.split(':')[0])
            temperature = get_weather_forecast(self.selected_date)
            
            if temperature is not None:
                predicted_passengers = int(self.predictor.predict(temperature, selected_hour))
                occupancy_level = self.get_occupancy_level(predicted_passengers)
                
                prediction_text.value = f"""
                📅 Datum: {self.selected_date.strftime('%Y-%m-%d')}
                🕒 Tid: {self.selected_time}
                🌡️ Temperatur: {temperature:.1f}°C
                👥 Förväntad beläggning: {predicted_passengers} passagerare
                📊 Nivå: {occupancy_level}
                """
            else:
                prediction_text.value = "❌ Kunde inte hämta väderprognos för valt datum"
            
            prediction_text.update()
    
    def get_occupancy_level(self, passengers):
        if passengers > 130:  # 80% av max
            return "🔴 Hög (Fullsatt)"
        elif passengers > 80:  # 50% av max
            return "🟡 Medel"
        else:
            return "🟢 Låg"

if __name__ == "__main__":
    app = BoatApp()
    ft.app(target=app.main)