# House Price Prediction Web Application

A complete Machine Learning-based House Price Prediction web application built with Django and Python. This application uses trained ML models to predict house prices based on various features like location, square footage, bedrooms, bathrooms, parking spaces, and year built.

## Features

### Machine Learning Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables (OneHotEncoder), and scales numeric features
- **Feature Engineering**: Derives **PropertyAge** from YearBuilt and supports optional advanced features (lot size, property type, condition, neighborhood rating, distance to CBD)
- **Model Training**: Trains multiple models (Ridge, Random Forest, HistGradientBoosting)
- **Model Validation**: Uses **5-fold cross-validation** + holdout testing
- **Best Model Selection**: Automatically selects and saves the best performing model
- **Model Persistence**: Saves a single end-to-end scikit-learn Pipeline using joblib (preprocessing + model)
- **Model Card + Metrics**: Saves `model_metrics.json` and `model_card.md` for reporting/defense

### Django Web Application Features
- **User Authentication**: Complete registration and login system
- **Prediction Form**: User-friendly form to input house details
- **Real-time Prediction**: Instant price prediction using trained ML model
- **Prediction History**: Stores and displays all user predictions
- **Export**: Download prediction history as CSV
- **User Profile**: Personal dashboard with statistics and recent predictions
- **Admin Dashboard**: Django admin interface for managing predictions
- **Chart Visualization**: Interactive charts showing price trends over time
- **Responsive Design**: Modern, mobile-responsive UI with Bootstrap 5

## Dataset Features

The ML model is trained on the following core features:
- **Location**: Categorical (Philippines-focused locations, e.g., Metro Manila, Cebu, Davao, etc.)
- **Floor Area (sqm)**: Numerical
- **Bedrooms**: Numerical (number of bedrooms)
- **Bathrooms**: Numerical (number of bathrooms)
- **Parking Spaces**: Numerical (number of parking spaces)
- **Year Built** → **PropertyAge**: Numerical (derived)

Optional advanced features (if available):
- Lot size (sqm), property type, house condition, neighborhood rating, distance to CBD (km)

**Target Output**: Predicted house price (PHP)

## Project Structure

```
house_price_project/
├── manage.py                      # Django management script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── house_price_project/           # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── predictions/                   # Django app for predictions
│   ├── __init__.py
│   ├── admin.py                   # Admin configuration
│   ├── apps.py                    # App configuration
│   ├── models.py                  # Database models
│   ├── views.py                   # View functions
│   ├── urls.py                    # URL routing
│   └── forms.py                   # Form classes
├── ml_training/                   # ML training scripts
│   ├── __init__.py
│   └── train_model.py             # Model training script
├── ml_models/                     # Saved ML models (created after training)
│   ├── model.pkl
│   └── model_metrics.json
├── templates/                     # HTML templates
│   └── predictions/
│       ├── base.html              # Base template
│       ├── home.html              # Home page
│       ├── predict.html           # Prediction form
│       ├── result.html            # Prediction result
│       ├── history.html           # Prediction history
│       ├── profile.html           # User profile
│       ├── login.html             # Login page
│       ├── register.html          # Registration page
│       └── logout.html            # Logout page
└── static/                        # Static files
    └── css/
        └── style.css              # Custom styles
```

## Installation Steps

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Navigate to Project
```bash
cd house_price_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the ML Model
```bash
python ml_training/train_model.py
```

This will:
- Load `data/house_prices.csv` if it exists (otherwise generate a sample dataset and save it there)
- Preprocess the data (impute missing values, encode Location, scale numeric features)
- Train Linear Regression and Decision Tree models
- Evaluate using R² and MAE
- Save the best model pipeline to `ml_models/model.pkl`

### Step 5: Run Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 6: Create Superuser (for Admin Access)
```bash
python manage.py createsuperuser
```

Follow the prompts to create an admin account.

### Step 7: Run the Development Server
```bash
python manage.py runserver
```

The application will be available at: `http://127.0.0.1:8000/`

Tip: you can also run the repo-level `start_server.py` script, which runs migrations automatically before starting the server.

## Usage

### 1. Register/Login
- Navigate to the home page
- Click "Register" to create a new account
- Or click "Login" if you already have an account

### 2. Predict House Price
- After logging in, click "Predict Price" in the navigation
- Fill in the house details:
  - Location (Philippines-focused locations)
  - Square feet (area size)
  - Number of bedrooms
  - Number of bathrooms
  - Parking spaces
  - Year built
- Click "Predict Price" to get the estimated price

### 3. View Prediction History
- Click "History" in the navigation
- View all your previous predictions in a table
- See a chart showing price trends over time

### 4. View Profile
- Click "Profile" in the navigation
- See your prediction statistics:
  - Total predictions
  - Average price
  - Highest and lowest prices
  - Recent predictions

### 5. Admin Dashboard
- Access the admin panel at `http://127.0.0.1:8000/admin/`
- Login with your superuser account
- Manage users and view all predictions

## Machine Learning Pipeline

The ML training script (`train_model.py`) follows this pipeline:

1. **Data Loading**: Loads a CSV with columns: Location, SquareFeet, Bedrooms, Bathrooms, ParkingSpaces, YearBuilt, Price
2. **Data Preprocessing**:
   - Missing values: imputed (most-frequent for Location, median for numeric)
   - Encoding: Location is one-hot encoded (unseen locations are handled safely)
   - Scaling: numeric features are standardized (primarily for Linear Regression)
3. **Train-Test Split**: 80% training / 20% testing
4. **Model Training**: Linear Regression and Decision Tree regression
5. **Model Evaluation**: R² and MAE
6. **Model Selection**: Picks the best model based on R²
7. **Model Persistence**: Saves a single Pipeline to `ml_models/model.pkl`

## Database Models

### User
Built-in Django User model with authentication support.

### PredictionHistory
Stores house price prediction records:
- `user`: ForeignKey to User model
- `location`: CharField (house location)
- `square_feet`: IntegerField (area size)
- `bedrooms`: IntegerField (number of bedrooms)
- `bathrooms`: IntegerField (number of bathrooms)
- `parking_spaces`: IntegerField (parking spaces)
- `year_built`: IntegerField (construction year)
- `predicted_price`: DecimalField (predicted price)
- `created_at`: DateTimeField (timestamp)

## Technologies Used

- **Backend**: Django 4.2.7
- **Machine Learning**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.3, numpy 1.26.2
- **Model Persistence**: joblib 1.3.2
- **Frontend**: Bootstrap 5.3.0
- **Charts**: Chart.js 4.4.0
- **Icons**: Font Awesome 6.4.0

## Customization

### Using Your Own Dataset
To use your own house price dataset:
1. Put your CSV at `data/house_prices.csv`, or pass `--data path/to/your.csv`
2. Ensure your dataset has the required columns: Location, SquareFeet, Bedrooms, Bathrooms, ParkingSpaces, YearBuilt, Price
   - Common variants like `location` or `square_feet` are accepted as well
3. Run the training script again:
   ```bash
   python ml_training/train_model.py --data data/house_prices.csv
   ```

### Adding More Features
To add more prediction features:
1. Add the feature to the ML training script
2. Update the `PredictionForm` in `forms.py`
3. Update the prediction form template
4. Update the model loading and prediction logic in `views.py`

### Changing UI Design
- Modify `static/css/style.css` for custom styling
- Update templates in `templates/predictions/` for layout changes

## Error Handling

The application includes:
- Form validation for user inputs
- Error messages for invalid data
- Exception handling in prediction logic
- User-friendly error messages

## Security Notes

- Change the `SECRET_KEY` in `settings.py` for production
- Set `DEBUG = False` in production
- Use environment variables for sensitive configuration
- Implement HTTPS in production
- Add CSRF protection (already enabled by Django)

## Future Enhancements

Potential improvements:
- Add more ML algorithms (XGBoost, LightGBM)
- Implement model retraining with new data
- Add image upload for house photos
- Include neighborhood crime rates
- Add school district information
- Implement user feedback on predictions
- Add export functionality for prediction history
- Create API endpoints for mobile apps

## License

This project is for educational purposes. Feel free to modify and use as needed.

## Support

For issues or questions, please refer to the code comments or Django/scikit-learn documentation.

---

**Happy Predicting! 🏠💰**
