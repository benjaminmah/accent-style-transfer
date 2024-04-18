import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import *
from model import *
import time
import math
import argparse
cuda = True if torch.cuda.is_available() else False


# Create an arg parser for the training process.
def training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-content', help='Content input')
    parser.add_argument('-content_weight', help='Content weight. Default is 1e2', default=1e2)
    parser.add_argument('-content_start', help='Time of the content audio to start reading at, in seconds. If unspecified, will start from the beginning.', default=None, type=float)
    parser.add_argument('-content_end', help='Time of the content audio to stop reading at, in seconds. If unspecified, will stop at the end.', default=None, type=float)
    parser.add_argument('-style', help='Style input')
    parser.add_argument('-style_weight', help='Style weight. Default is 1', default=1)
    parser.add_argument('-style_start', help='Time of the style audio to start reading at, in seconds. If unspecified, will start from the beginning.', default=None, type=float)
    parser.add_argument('-style_end', help='Time of the style audio to stop reading at, in seconds. If unspecified, will stop at the end.', default=None, type=float)
    parser.add_argument('-epochs', type=int, help='Number of epoch iterations. Default is 20000', default=5000) # was 20000
    parser.add_argument('-print_interval', type=int, help='Number of epoch iterations between printing losses', default=1000)
    parser.add_argument('-plot_interval', type=int, help='Number of epoch iterations between plot points', default=1000)
    parser.add_argument('-learning_rate', type=float, default=0.002)
    parser.add_argument('-output', help='Output file name. Default is "output"', default='output')
    return parser


# Helper function to measure elapsed time since a given time stamp.
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Overall training function.
def train_model(args):
    CONTENT_FILENAME = args.content
    STYLE_FILENAME = args.style

    a_content, sr = audio_to_spectrum(CONTENT_FILENAME, offset=args.content_start, duration=args.content_end - args.content_start)
    a_style, sr = audio_to_spectrum(STYLE_FILENAME, offset=args.style_start, duration=args.style_end - args.style_start)

    a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
    if cuda:
        a_content_torch = a_content_torch.cuda()
    print(a_content_torch.shape)
    a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
    if cuda:
        a_style_torch = a_style_torch.cuda()
    print(a_style_torch.shape)

    model = RandomCNN()
    model.eval()

    a_C_var = Variable(a_content_torch, requires_grad=False).float()
    a_S_var = Variable(a_style_torch, requires_grad=False).float()
    if cuda:
        model = model.cuda()
        a_C_var = a_C_var.cuda()
        a_S_var = a_S_var.cuda()

    a_C = model(a_C_var)
    a_S = model(a_S_var)

    # Optimizer
    learning_rate = args.learning_rate
    a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3)
    if cuda:
        a_G_var = a_G_var.cuda()
    a_G_var.requires_grad = True
    optimizer = torch.optim.Adam([a_G_var])

    # Coefficient of content and style
    style_param = args.style_weight
    content_param = args.content_weight

    num_epochs = args.epochs
    print_every = args.print_interval
    plot_every = args.plot_interval

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()
    # Train the model
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        a_G = model(a_G_var)

        content_loss = content_param * compute_content_loss(a_C, a_G)
        style_loss = style_param * compute_layer_style_loss(a_S, a_G)
        loss = content_loss + style_loss

        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Print
        if epoch % print_every == 0:
            print("{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                        epoch / num_epochs * 100,
                                                                                        time_since(start),
                                                                                        content_loss.item(),
                                                                                        style_loss.item(), loss.item()))
            current_loss += loss.item()

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
    gen_audio_C = args.output + ".wav"
    spectrum_to_wav(gen_spectrum, sr, gen_audio_C)

    plt.figure()
    plt.plot(all_losses)
    plt.savefig(args.output + '_loss_curve.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Content Spectrum")
    plt.imsave(args.output + '_content_spectrum.png', a_content[:400, :])
    plt.close()

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Style Spectrum")
    plt.imsave(args.output + '_style_spectrum.png', a_style[:400, :])
    plt.close()

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("CNN Voice Transfer Result")
    plt.imsave(args.output + '_gen_spectrum.png', gen_spectrum[:400, :])
    plt.close()


if __name__ == "__main__":
    parser = training_parser()
    args = parser.parse_args()

    train_model(args)