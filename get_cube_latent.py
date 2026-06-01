import torch

from core import  *
if __name__ == "__main__":
    model = VAE()
    model.eval()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_path = 'data'
    groups = acdc_patients_group(data_path)
    group = groups['01.012.OS']
    img_sequence, volume_sequence = read_mats_tensor(group)
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize((0.5,), (0.5,))
    ])

    T, S, H, W = img_sequence.shape
    res = []
    for t in range(T):
        cube_res = []
        for s in range(S):
            x = img_sequence[t, s].numpy()
            x = transform(image=x)['image']
            x = transforms.ToTensor()(x)

            x = x.repeat(1, 3, 1, 1)
            x = x.to(device)
            with torch.no_grad():
                z = model.encode(x)
                print(z.shape)
                cube_res.append(z.squeeze())
                # np.save(os.path.join(cube_dir, f'{cube_name}_{i}.npy'), z.cpu().numpy())
                xrec = model.decode(z)
                xrec = torch.mean(xrec, dim=1, keepdim=True)

                print(x.max(), x.min())
                # plt.imshow(x.cpu().numpy()[0, 0])
                # plt.show()
                #
                # plt.imshow(xrec.cpu().numpy()[0, 0])
                # plt.show()
        cube_res = torch.stack(cube_res)
        res.append(cube_res)

    res = torch.stack(res).cpu().numpy()
    print(res.shape)
    np.save(os.path.join('latent_data', f'{"01.012.OS"}.npy'), res)