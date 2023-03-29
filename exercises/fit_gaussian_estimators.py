import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    MU, SIGMA = 10,1
    SAMPLE_SIZE = 1000
    X : np.ndarray = np.random.normal(MU, SIGMA, SAMPLE_SIZE)
    fitted_model = UnivariateGaussian().fit(X)
    print("Question 1:")
    print("____________________\n")
    print(f"({fitted_model.mu_},{fitted_model.var_})")
    print("\n")


    # Question 2 - Empirically showing sample mean is consistent
    estimated_mu = [np.abs(MU - UnivariateGaussian().fit(X[0:10*n]).mu_) for n in range(1,SAMPLE_SIZE//10 + 1)]
    estimated_mu_df = pd.DataFrame(np.array([estimated_mu,10*np.arange(1,101,1)]).T,columns=["Mean Estimator Deviation","Sample Size"])
    fig = px.scatter(estimated_mu_df,x="Sample Size", y="Mean Estimator Deviation",
               title="Mean Estimator Deviation As a Function of Sample Size",width=600,height=600)
    fig.update_yaxes(range=[-0.01,0.8])
    # fig.show()
    # fig.write_image("normal_dist_mean_deviation.png") ## i obviously used this to save the image but i hid it
    # to ease the grading proccess in case it will be run by you guys

    # Question 3 - Plotting Empirical PDF of fitted model
    PDF = np.array([X,fitted_model.pdf(X)]).T
    indexes = np.argsort(PDF[:,0])
    PDF = PDF[indexes]
    PDF_DF = pd.DataFrame(PDF,columns=["x","cols"])
    fig1_3 = px.scatter(PDF_DF,x="x",y="cols")
    fig1_3.update_layout(yaxis_title={"text":r'$f_{\hat{\mathcal{N}}}(x)$',"font": {"size" : 15}},
                         title={"text": "Empirical PDF Function Under the Fitted Model",
                                "xanchor" : "center", "yanchor" : "top", "x" : 0.5, "y" : 0.95},
                         xaxis_title = {"font" : {"size" : 15}})
    # fig1_3.show()

    fig1_3.write_image("PDF_model.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    MU = np.array([0,0,4,0])
    SIGMA = np.array([[1,0.2,0,0.5],
                      [0.2,2,0,0],
                     [0,0,1,0],
                     [0.5,0,0,1]])
    SAMPLE_SIZE = 1000
    X = np.random.multivariate_normal(MU,SIGMA,SAMPLE_SIZE)
    fitted_model = MultivariateGaussian().fit(X)
    print(" Question 4 ")
    print("fitted model Mu:")
    print(fitted_model.mu_)
    print("\n")
    print("fitted model Cov:")
    print(fitted_model.cov_)
    print("\n")


    # Question 5 - Likelihood evaluation
    z_ : np.ndarray= np.zeros((200,200))
    f_vals = np.linspace(-10,10,200)
    for i,f_1 in enumerate(f_vals):
        for j,f_3 in enumerate(f_vals):
            curr_mu = np.array([f_1,0,f_3,0])
            z_[i,j] = MultivariateGaussian.log_likelihood(curr_mu,SIGMA,X)

    heatmap = go.Figure(go.Heatmap(x = f_vals,y=f_vals,z=z_))
    heatmap.update_layout(title = {"text": r"Log-Likelihood as a Function of Expected Value of Features 1 and 3",
                                   "xanchor" : "center", "yanchor" : "top", "x" : 0.5, "y" : 0.95},
                          yaxis_title = {"text":r'$f_1$',"font": {"size" : 15}},
                          xaxis_title = {"text":r'$f_3$',"font": {"size" : 15}})
    heatmap.write_image("heatmap.png")


    # Question 6 - Maximum likelihood
    idx = np.unravel_index(z_.argmax(),z_.shape)
    print("Question 6")
    print("log-likelihood argmax")
    print()
    print(np.round(f_vals[idx[0]],3),np.round(f_vals[idx[1]],3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
