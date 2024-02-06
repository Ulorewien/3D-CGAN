import matplotlib.pyplot as plt

folder = "D:\Studies\Sem8\Project\\3DCGAN"
model_types = ["disc_simple", "disc_complex", "sphere_simple", "sphere_complex"]
model_names = ["Disc Simple", "Disc Complex", "Sphere Simple", "Sphere Complex"]

def get_losses(data):
    d_loss_real, d_loss_fake, g_loss = [], [], []
    for i in range(0, len(data), 2):
        line = data[i]
        if(line[24:35] == "d_loss_real"):
            d_loss_real.append(float(line[36:41]))
            d_loss_fake.append(float(line[54:59]))
            g_loss.append(float(line[67:72]))
        elif(line[25:36] == "d_loss_real"):
            d_loss_real.append(float(line[37:42]))
            d_loss_fake.append(float(line[55:60]))
            g_loss.append(float(line[68:73]))
        elif(line[26:37] == "d_loss_real"):
            d_loss_real.append(float(line[38:43]))
            d_loss_fake.append(float(line[56:61]))
            g_loss.append(float(line[69:74]))
        elif(line[27:38] == "d_loss_real"):
            d_loss_real.append(float(line[39:44]))
            d_loss_fake.append(float(line[57:62]))
            g_loss.append(float(line[70:75]))
    return d_loss_real, d_loss_fake, g_loss

def plot_losses(d_loss_real, d_loss_fake, g_loss, model_name):
    plt.plot(d_loss_real, label="D_loss_real")
    plt.plot(d_loss_fake, label="D_loss_fake")
    plt.plot(g_loss, label="G_loss")
    plt.title(model_name + " (GAN losses v Iterations)")
    plt.xlabel("No. of iterations")
    plt.ylabel("Loss value")
    plt.legend()
    plt.show()

for i in range(len(model_types)):
    model_type = model_types[i]
    model_name = model_names[i]
    data = ""
    n_lines = 0

    with open(folder+"\\"+model_type+"_log.txt", "r") as file:
        data = file.read()
        data = data.split("\n")

    d_loss_real, d_loss_fake, g_loss = get_losses(data)
    plot_losses(d_loss_real, d_loss_fake, g_loss, model_name)

    break