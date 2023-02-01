import logging
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch.nn as nn

debug = False


loss_i = 0
def custom_loss(output, target, args):
    n_element = output.numel()

    # MSE
    l1_loss = F.l1_loss(output, target)
    l1_loss *= args.loss_l1_weight

    # continuous motion
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss *= args.loss_cont_weight

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= args.loss_var_weight

    loss = l1_loss + cont_loss + var_loss

    # inspect loss terms
    global loss_i
    if loss_i == 100:
        logging.debug('  (loss terms) l1 %.5f, cont %.5f, var %.5f' % (l1_loss.item(), cont_loss.item(), var_loss.item()))
        loss_i = 0
    loss_i += 1

    return loss


def train_iter_seq2seq(args, epoch, in_text, in_lengths, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    outputs = net(in_text, in_lengths, target_poses, None)

    # loss
    loss = custom_loss(outputs, target_poses, args)
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def train_iter_DAE(args, epoch, noisy_poses, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()


    # generation
    if args.autoencoder_vq=='True' and args.autoencoder_vae == 'False':
        outputs, vq_loss, perplexity_vq = net(noisy_poses)
    elif args.autoencoder_vq == 'True' and args.autoencoder_vae == 'True':
        outputs, vq_loss, perplexity_vq, logvar, meu = net(noisy_poses)
    elif args.autoencoder_vq == 'False' and args.autoencoder_vae == 'True':
        outputs, logvar, meu = net(noisy_poses)
    else:
        outputs = net(noisy_poses)

    # loss
    loss_fn = torch.nn.MSELoss()

    rec_loss = loss_fn(outputs, target_poses)

    if args.autoencoder_vq == 'True':
        GSOFT = False
        if GSOFT:
            rec_loss = outputs.log_prob(target_poses).sum(dim=1).mean()


            loss = vq_loss - rec_loss/100
            # print("LOSSSSSSSS!", vq_loss, rec_loss)
        else:
            loss = rec_loss + vq_loss
    else:
        loss = rec_loss

    if args.autoencoder_vae == 'True':
        loss_KLD = -2.5 * torch.mean(torch.mean(1 + logvar - logvar.exp() - meu.pow(2), 1))
        # loss_KLD = -0.5 * torch.mean(torch.mean(1 + logvar - logvar.exp(), 1))
        print("Kista", loss_KLD)
        loss += 5 * loss_KLD #0.11

    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    if args.autoencoder_vq == 'True':
        return {'loss': rec_loss.item()}, perplexity_vq
    else:
        return {'loss': loss.item()}



def train_iter_Autoencoder_seq2seq(args, epoch, input_poses, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    if args.autoencoder_vae == 'True':
        outputs, _, meu, logvar = net(input_poses, target_poses)
    else:
        outputs , _ = net(input_poses, target_poses)

    # loss
    #Todo: important: I removed custom loss and replaced it by ll to test
    loss = custom_loss(outputs, target_poses, args)
    # loss  = F.mse_loss(outputs, target_poses)

    if args.autoencoder_vae == 'True':

        # loss_KLD = 0.5 * torch.mean(logvar.exp()-logvar-1 + meu.pow(2))
        loss_KLD = -0.5 * torch.mean(torch.mean( 1  + logvar - logvar.exp() - meu.pow(2), 1))
        # if epoch%10==0:
        #     print("____________________")
        #     print("loss", loss)
        #     print("loss_KLD", loss_KLD )
        #     print("Epoch ratio",epoch, args.epochs, epoch/args.epochs)
        #     print("____________________")

        kl_start_epoch = 5
        if epoch>kl_start_epoch and args.autoencoder_freeze_encoder == 'False':
            loss += loss_KLD * 0.01 * (epoch-kl_start_epoch)/args.epochs
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}


def train_iter_Autoencoder_ssl_seq2seq(args, epoch, input_poses, target_poses, net, optim,
                                       stack_pairs1, stackpairs2, stack_label):
    # zero gradients
    optim.zero_grad()

    # generation
    # Unlabeled
    if args.autoencoder_vae == 'True':
        outputs, _, meu, logvar = net(input_poses, target_poses)
        loss_KLD = -0.5 * torch.mean(torch.mean(1 + logvar - logvar.exp() - meu.pow(2), 1))

    else:
        outputs, _ = net(input_poses, target_poses)

    # labeled
    if debug:
        print("stack_pairs1", stack_pairs1.shape)
        print("net.decoder.n_layers", net.decoder.n_layers)
    if args.autoencoder_vae == 'True':
        outputs_p1, latents_p1, mu_1, logvar_1 = net(stack_pairs1, stack_pairs1)
        outputs_p2, latents_p2, mu_2, logvar_2 = net(stackpairs2, stackpairs2)
    else:
        outputs_p1, latents_p1 = net(stack_pairs1, stack_pairs1)
        outputs_p2, latents_p2 = net(stackpairs2, stackpairs2)

    if debug:
        print("1. latentp1.shape:", latents_p1.shape)
    latents_p1 = torch.hstack((latents_p1[0], latents_p1[1]))
    if debug:
        print("2. latentp1.shape:", latents_p1.shape)

    latents_p2 = torch.hstack((latents_p2[0], latents_p2[1]))
    # Normal loss
    #Todo: important: I removed custom loss and replaced it by ll to test
    # loss = custom_loss(outputs, target_poses, args)
    # loss_unlabeled = F.mse_loss(outputs, target_poses)
    loss_unlabeled = custom_loss(outputs, target_poses, args)

    # Pairwise loss
    # cos_dist = [torch.nn.functional.cosine_similarity(stack_pairs1[n, n, :], output[:, n - 1, :]) for n in range(1, output.shape[1])]
    cos_dist = F.cosine_similarity(latents_p1, latents_p2)
    if debug:
        print("latentp1.shape:", latents_p1.shape)
        print("cosine_similarity.shape:", cos_dist.shape)
        print("stack_label", stack_label.shape)
    mask = (stack_label == 1)
    cos_dist[mask] = cos_dist[mask]*-1
    loss_labeled = torch.sum(cos_dist)

    loss = args.loss_label_weight + loss_unlabeled

    if args.autoencoder_vae == 'True':
        kl_start_epoch = 10
        if epoch > kl_start_epoch:
            loss += loss_KLD * 0.1 * (epoch - kl_start_epoch) / args.epochs


    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()
    if debug:
        print("loss_unlabeled", loss_unlabeled)
        print("Loss_labeled", loss_labeled)
        print("loss_label_weight:", args.loss_label_weight)
    return {'loss': loss.item()}






def train_iter_c2g_seq2seq(args, epoch, input_Cluster, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    outputs = net(input_Cluster, target_poses)

    # loss
    #Todo: important: I removed custom loss and replaced it by ll to test
    loss = custom_loss(outputs, target_poses, args)
    # loss  = F.mse_loss(outputs, target_poses)
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}


def train_iter_text2embedding(args, epoch, in_text, in_lengths, in_audio, target_poses, cluster_targets,
                              GPT3_Embedding, net, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    if args.text2_embedding_discrete == 'False':
        outputs, _ = net(in_text, in_lengths, in_audio, target_poses, GPT3_Embedding, None)
    else:
        outputs, _ = net(in_text, in_lengths, in_audio, cluster_targets, GPT3_Embedding, None)
    # loss
    # print(outputs.shape)
    if args.text2_embedding_discrete=='False':
        loss = F.mse_loss(outputs[:, 1:, :], target_poses[:, 1:, :])
    else:
        os = cluster_targets.shape
        if debug:
            print("cc", cluster_targets.shape)
            q = cluster_targets.reshape(-1)
            print(q)
            print("cc", q.shape)
            w = F.one_hot(q.to(torch.int64), 300)
            print("----", w.shape, w)


        # cluster_targets_one_hot = F.one_hot(cluster_targets.reshape(-1).to(torch.int64), 300)
        # cluster_targets_one_hot = cluster_targets_one_hot.reshape(os[0], os[1], -1)
        outputs = outputs[:, 1:, :]
        outputs = outputs.reshape(-1, outputs.shape[2])
        # cluster_targets.reshape(-1)

        cluster_targets = cluster_targets[:,1:]
        cluster_targets = cluster_targets.reshape(-1)

        if debug:
            print("cluster_targets_one_hot", cluster_targets.shape)
            print("outputs", outputs.shape)
        a = outputs.cpu().detach().numpy()
        b = cluster_targets.cpu().detach().numpy()
        loss = torch.nn.CrossEntropyLoss()(outputs.float(), cluster_targets.long())
        # loss = torch.nn.NLLLoss()(outputs.float(), cluster_targets.long())

        # cluster_targets_one_hot = F.one_hot(cluster_targets.reshape(-1).to(torch.int64), 514)
        # loss = F.l1_loss(outputs, cluster_targets_one_hot)

    # loss2 = torch.mean(torch.mean(loss, dim=2)/cluster_portion)





    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item()}


def train_iter_text2embedding_GAN(args, epoch, in_text, in_lengths, target_poses, cluster_portion,
                                  g_net, d_net, g_optim, d_optim):
    # zero gradients
    bce_loss = torch.nn.BCELoss()

    # generation
    # 1. * Generate fake data

    with torch.no_grad():
        fake_y = g_net(in_text, in_lengths,  target_poses, None)

    # 2. * Train Discriminator

    d_optim.zero_grad()
    g_optim.zero_grad()
    d_real_error = 0
    d_fake_error = 0

    real_logit = d_net(in_text, in_lengths, target_poses, None )
    real_label = torch.ones_like(real_logit)
    real_error = bce_loss(real_logit, real_label)
    d_real_error = torch.mean(real_error)

    fake_logit = d_net(in_text, in_lengths, fake_y, None )
    fake_label = torch.zeros_like(fake_logit)
    if debug:
        print("fake_label", fake_label.shape)
        print("fake_label", fake_label)
    fake_error = bce_loss(fake_logit, fake_label)
    d_fake_error = torch.mean(fake_error)


    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optim.step()

    d_real_loss = (d_real_error.cpu().detach().numpy())
    d_fake_loss = (d_fake_error.cpu().detach().numpy())

    # 2. * Unrolling step
    unroll_steps = 10
    if unroll_steps:
        # * Unroll D
        d_backup = d_net.state_dict()
        for k in range(unroll_steps):
            # * Train D
            d_optim.zero_grad()
            d_real_error = 0
            d_fake_error = 0


            real_logit = d_net(in_text, in_lengths, target_poses, None )
            real_label = torch.ones_like(real_logit)
            real_error = bce_loss(real_logit, real_label)
            d_real_error = torch.mean(real_error)

            fake_logit = d_net(in_text, in_lengths, fake_y, None)
            fake_label = torch.zeros_like(fake_logit)
            fake_error = bce_loss(fake_logit, fake_label)
            d_fake_error = torch.mean(fake_error)

            d_loss = d_real_error + d_fake_error
            d_loss.backward()
            d_optim.step()

    # * Train G
    g_optim.zero_grad()

    gen_y = g_net(in_text, in_lengths, target_poses, None )

    gen_logit = d_net(in_text, in_lengths, gen_y, None)
    gen_lable = torch.ones_like(gen_logit)
    gen_error = bce_loss(gen_logit, gen_lable)
    g_error = torch.mean(gen_error)


    g_error.backward()
    g_optim.step()

    if unroll_steps:
        d_net.load_state_dict(d_backup)

    # Todo: is it necessary
    # torch.nn.utils.clip_grad_norm_(g_net.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(d_net.parameters(), 5)

    # # loss
    # # print(outputs.shape)
    # loss = F.mse_loss(outputs, target_poses)
    # # loss2 = torch.mean(torch.mean(loss, dim=2)/cluster_portion)
    # loss.backward()
    # # optimize
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    # optim.step()

    # {'loss': loss.item()},
    return d_real_loss, d_fake_loss




def train_iter_Autoencoder_VQ_seq2seq(args, epoch, input_poses, target_poses, net, optim):
    # zero gradients
    optim.zero_grad()
    vq_start_epoch = 0


    # net.vq_layer.embedding_grad(epoch % 3 == 0)

    # generation
    if args.autoencoder_vq == 'True' and args.autoencoder_vae == 'True':
        outputs, _, meu, logvar, loss_vq, perplexity_vq = net(input_poses, target_poses, epoch > vq_start_epoch)
    if args.autoencoder_vq == 'True' and args.autoencoder_vae == 'False':
        outputs, _, loss_vq, perplexity_vq = net(input_poses, target_poses, epoch > vq_start_epoch)
    if args.autoencoder_vq == 'False' and args.autoencoder_vae == 'True':
            outputs, _, meu, logvar = net(input_poses, target_poses)
    if args.autoencoder_vq == 'False' and args.autoencoder_vae == 'False':
        outputs , _ = net(input_poses, target_poses)

    # loss
    #Todo: important: I removed custom loss and replaced it by ll to test
    loss = custom_loss(outputs, target_poses, args)
    # loss  = F.mse_loss(outputs, target_poses)

    if args.autoencoder_vae == 'True':
        # loss_KLD = 0.5 * torch.mean(logvar.exp()-logvar-1 + meu.pow(2))
        loss_KLD = -0.5 * torch.mean(torch.mean( 1  + logvar - logvar.exp() - meu.pow(2), 1))
        loss_KLD = 0.5 * torch.mean(logvar.exp() - logvar - 1 + meu.pow(2))
        if epoch%10==0:
            print("____________________")
            print("loss", loss)
            print("loss_KLD", loss_KLD )
            print("Epoch ratio",epoch, args.epochs, epoch/args.epochs)
            print("____________________")
        if debug:
            print('rec_loss', loss)
            # print("loss_vqn", loss_vq)
            print("loss_KLD", loss_KLD)

        kl_start_epoch = 0
        if epoch>kl_start_epoch:
            loss += loss_KLD * 0.1 * (epoch-kl_start_epoch)/args.epochs

    if epoch > vq_start_epoch:
        if args.autoencoder_vq == 'True':
            emb_w = net.vq_layer._embedding.weight.detach()
            # mycdist = torch.cdist(emb_w, emb_w).mean()
            # print("________________________________________________\n loss:",
            #       loss.data, "  vq_los", loss_vq.data, "mycdist:", mycdist.data)

            loss = (loss) + 1 * loss_vq/400 #+ -1*torch.log(mycdist)



    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    # if epoch % 1 == 0:
    #     print("____________________")
    #     print("loss", loss)
    #     # print("loss_KLD", loss_KLD)
    #     print("Loss:", loss, "-------Loss_VQ", loss_vq)
    #     print("____________________")


    if args.autoencoder_vq == 'True':
        # return {'loss': loss.item()}, loss_vq
        # Todo: it should be perplexity_vq not loss_vq
        return {'loss': loss.item()}, perplexity_vq.detach()
    else:
        return {'loss': loss.item()}






