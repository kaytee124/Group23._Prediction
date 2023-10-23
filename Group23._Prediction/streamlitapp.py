import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model from the downloaded pickle file
with open('gb_regressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Load the scaler from the .pkl file
with open('scaler.pkl', 'rb') as scaler_file:
    sc = joblib.load(scaler_file)


# Create a Streamlit app
st.title("Machine Learning Model Deployment")

# Add input elements for user interaction
feature1 = st.number_input("sofifa_id")
feature2 = st.number_input("potential")
feature3 = st.number_input("value_eur")
feature4 = st.number_input("wage_eurl")
feature5 = st.number_input("age")
feature6 = st.number_input("international_reputation")
feature7 = st.number_input("release_clause_eur")
feature8 = st.number_input("passing")
feature9 = st.number_input("dribbling")
feature10 = st.number_input("physic")
feature11 = st.number_input("attacking_crossing")
feature12= st.number_input("attacking_short_passing")
feature13= st.number_input("skill_curve")
feature14= st.number_input("skill_long_passing")
feature15= st.number_input("skill_ball_control")
feature16= st.number_input("movement_reactions")
feature17= st.number_input("power_shot_power")
feature18= st.number_input("power_long_shots")
feature19= st.number_input("mentality_aggression")
feature20= st.number_input("mentality_vision")
feature21= st.number_input("mentality_composure")
feature22= st.text_input("ls")
feature23= st.text_input("st")
feature24= st.text_input("rs")
feature25= st.text_input("lw")
feature26= st.text_input("lf")
feature27= st.text_input("cf")
feature28= st.text_input("rf")
feature29= st.text_input("rw")
feature30= st.text_input("lam")
feature31= st.text_input("cam")
feature32= st.text_input("ram")
feature33= st.text_input("lm")
feature34= st.text_input("lcm")
feature35= st.text_input("cm")
feature36= st.text_input("rcm")
feature37= st.text_input("rm")
feature38= st.text_input("lwb")
feature39= st.text_input("ldm")
feature40= st.text_input("cdm")
feature41= st.text_input("rdm")
feature42= st.text_input("rwb")
feature43= st.text_input("lb")
feature44= st.text_input("lcb")
feature45= st.text_input("cb")
feature46= st.text_input("rcb")
feature47= st.text_input("rb")
feature48= st.text_input("gk")
feature49= st.number_input("shooting ")


feature1 = float(feature1) 
feature2 = float(feature2) 
feature3 = float(feature3) 
feature4 = float(feature4) 
feature5 =float(feature5) 
feature6 =float(feature6) 
feature7 =float(feature7)
feature8 =float(feature8)  
feature9 =float(feature9) 
feature10 = float(feature10) 
feature11 =float(feature11) 
feature12 =float(feature12) 
feature13 =float(feature13) 
feature14=float(feature14) 
feature15=float(feature15) 
feature16 =float(feature16) 
feature17 =float(feature17) 
feature18 =float(feature18) 
feature19 =float(feature19) 
feature20 =float(feature20) 
feature21=float(feature21)
feature49 = float(feature49)
feature22 = str(feature22)
feature23 = str(feature23)
feature24 = str(feature24)
feature25 = str(feature25)
feature26 = str(feature26)
feature27 = str(feature27)
feature28 = str(feature28)
feature29 = str(feature29)
feature30 = str(feature30)
feature31 = str(feature32)
feature33 = str(feature33)
feature34 = str(feature34)
feature35 = str(feature35)
feature36 = str(feature36)
feature37 = str(feature37)
feature38 = str(feature38)
feature39 = str(feature39)
feature40 = str(feature40)
feature41 = str(feature41)
feature42 = str(feature42)
feature43 = str(feature43)
feature44= str(feature44)
feature45 = str(feature45)
feature46 = str(feature46)
feature47 = str(feature47)
feature48 = str(feature48)





feature = []
feature.append(feature1)
feature.append(feature2)
feature.append(feature3)
feature.append(feature4)
feature.append(feature5)
feature.append(feature6)
feature.append(feature7)
feature.append(feature8)
feature.append(feature9)
feature.append(feature10)
feature.append(feature11)
feature.append(feature12)
feature.append(feature13)
feature.append(feature14)
feature.append(feature15)
feature.append(feature16)
feature.append(feature17)
feature.append(feature18)
feature.append(feature19)
feature.append(feature20)
feature.append(feature21)
feature.append(feature49)

nonnumfeature =[]
nonnumfeature.append(feature22)
nonnumfeature.append(feature23)
nonnumfeature.append(feature24)
nonnumfeature.append(feature25)
nonnumfeature.append(feature26)
nonnumfeature.append(feature27)
nonnumfeature.append(feature28)
nonnumfeature.append(feature29)
nonnumfeature.append(feature30)
nonnumfeature.append(feature31)
nonnumfeature.append(feature32)
nonnumfeature.append(feature33)
nonnumfeature.append(feature34)
nonnumfeature.append(feature35)
nonnumfeature.append(feature36)
nonnumfeature.append(feature37)
nonnumfeature.append(feature38)
nonnumfeature.append(feature39)
nonnumfeature.append(feature40)
nonnumfeature.append(feature41)
nonnumfeature.append(feature42)
nonnumfeature.append(feature43)
nonnumfeature.append(feature44)
nonnumfeature.append(feature45)
nonnumfeature.append(feature46)
nonnumfeature.append(feature47)
nonnumfeature.append(feature48)






dffeature = pd.DataFrame(feature)
dfnonnumfeature = pd.DataFrame(nonnumfeature)


dffeature= dffeature.T

label_encoder = LabelEncoder()

for column in dfnonnumfeature.columns:
        dfnonnumfeature[column] = label_encoder.fit_transform(dfnonnumfeature[column])

dfnonnumfeature = dfnonnumfeature.T

dfnonnumfeature.columns = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram",
    "lm", "lcm", "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb",
    "lb", "lcb", "cb", "rcb", "rb", "gk"
]


dffeature.columns= [
    "sofifa_id", "potential", "value_eur", "wage_eur", "age", "international_reputation",
    "release_clause_eur", "shooting", "passing", "dribbling", "physic", "attacking_crossing", "attacking_short_passing",
    "skill_curve", "skill_long_passing", "skill_ball_control", "movement_reactions", "power_shot_power",
    "power_long_shots", "mentality_aggression", "mentality_vision", "mentality_composure"
]




newfeature = pd.concat([dffeature, dfnonnumfeature], axis=1)



scaled = sc.transform(newfeature)
newfeature=pd.DataFrame(scaled, columns=newfeature.columns)

if st.button("Predict"):

    


    # Create an empty list to store multiple predictions
    predictions = []

    # Number of Monte Carlo simulations for estimating prediction intervals
    num_simulations = 100  # You can adjust this as needed

    for _ in range(num_simulations):
        # Perturb the input features slightly to simulate variations
        perturbed_feature = newfeature.copy()  # Copy the scaled features
        perturbed_feature += np.random.normal(0, 0.1, size=perturbed_feature.shape)  # Add some noise

        # Make predictions for the perturbed feature
        prediction = model.predict(perturbed_feature)
        predictions.append(prediction[0])

    # Calculate the standard deviation of the predictions
    std_dev = np.std(predictions)

    # Display the prediction and confidence (standard deviation)
    st.write(f"Predicted Output: {np.mean(predictions)}")
    st.write(f"Prediction Confidence : {std_dev:.4f}")
