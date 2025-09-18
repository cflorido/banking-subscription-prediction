# Banking Products – Customer Subscription Prediction

This project aimed to analyze customer behavior in relation to the subscription of term deposits (CDT), with the objective of identifying key influencing factors and designing strategies that maximize the bank’s revenue. The work followed a complete data science and machine learning pipeline, from business question formulation and exploratory data analysis to the development of predictive models and deployment of an interactive dashboard.

The preprocessing stage included validation of duplicates and missing values, categorical consistency checks, and outlier detection. Outliers in variables such as **balance** and **call duration** were retained due to their business relevance. Categorical variables were encoded using integer and binary encoding, and numerical features were normalized to improve model performance.  

Exploratory data analysis revealed that **call duration**, **balance**, and **age** were the most significant predictors of customer subscription, with seasonal effects across months. Visualization techniques such as correlation matrices, bar plots, and scatter plots provided insights for customer segmentation and marketing strategies.  

Several algorithms were tested, including **Random Forest** and **Neural Networks**, with additional balancing techniques such as **SMOTEENN** to address class imbalance. An analytical revenue function was defined based on the outcomes of the confusion matrix. Threshold optimization was performed to evaluate trade-offs between accuracy and expected revenue, and the adoption of a **0.3 threshold** was recommended as it maximizes financial returns.  

The final solution was implemented as an interactive dashboard that enables business users to:  
- Explore the impact of different thresholds on both accuracy and revenue.  
- Visualize classification results and revenue decomposition by confusion matrix outcomes.  
- Query customer profiles and review subscription probabilities.  

The dashboard was deployed on **AWS (E2 Large instance)** to ensure accessibility and scalability, providing the commercial area with an intuitive tool for decision-making. By combining predictive analytics with interactive visualization, the system supports evidence-based strategies to increase customer acquisition and maximize bank revenue.  

---
## Documents
- [Project Report – English](https://github.com/user-attachments/files/22415982/Proyecto2-Analitica.1.1.pdf)  
- [Reporte del Proyecto – Español](https://github.com/user-attachments/files/22415961/Proyecto2-Analitica.pdf)  
---
## Presentations
- [Presentation – English](https://github.com/user-attachments/files/22415984/Banco.1.1.pdf)  
- [Presentación – Español](https://github.com/user-attachments/files/22415962/Banco.1.pdf)  

---

## Authors
- Carol Sofía Florido Castro – 202111430  
- Natalia Villegas Calderón – 202113370  
- Alejandra Garzón Carvajal – 202116863  

Course: Business Intelligence – ISIS 3301  
City: Bogotá, Colombia  
Year: 2025  

---

## Repository Structure
```bash
Fase1/
│── BusinessQuestions.pdf
│── ExploratoryAnalysis.ipynb
│── DataPreparation.ipynb

Fase2/
│── Modeling.ipynb
│── ThresholdAnalysis.ipynb
│── dashboard.py
│── requirements.txt
```


---

## Technologies Used
- Python 3.11  
- scikit-learn, pandas, numpy, matplotlib, seaborn  
- imbalanced-learn (SMOTEENN)  
- TensorFlow / Keras (Neural Networks)  
- Dash / Plotly (Dashboard development)  
- AWS (Deployment)  

---

## How to Run
1. Install dependencies  
   ```bash
   pip install -r requirements.txt

2. Run the dashboard
  ```bash
  python app.py
  ```

4. Access at
  ```bash
  http://127.0.0.1:8050
   ```
