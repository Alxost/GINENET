class GINENet(torch.nn.Module):
    def __init__(self):
        super(GINENet, self).__init__()
        torch.manual_seed(12345)
        dim = 64
        self.atomicNumEmb = nn.Embedding(10,5)
        self.formalChargeEmb = nn.Embedding(12,6)
        self.aromaticEmb = nn.Embedding(2,1)
        self.valenceEmb = nn.Embedding(9,4)
        self.hybridizationEmb = nn.Embedding(6,3)
        self.chiralityEmb = nn.Embedding(6,3)
        self.numHEmb = nn.Embedding(10,5)
        self.degreeEmb = nn.Embedding(12,6)
        self.typeEmb = nn.Embedding(2,1)
        self.residueEmb = nn.Embedding(31, 14)

        self.bondTypeEmb = nn.Embedding(7,3)
        self.bondDirEmb = nn.Embedding(5,2)
        self.stereoEmb = nn.Embedding(6,3)
        self.inRingEmb = nn.Embedding(2,1)
        self.innerEmb = nn.Embedding(2,1)

        self.conv1 = GINEConv(Sequential(Linear(49, dim), BatchNorm(dim), ReLU(),
                                         Linear(dim, dim), ReLU()),
                                         edge_dim = 11)

        self.conv2 = GINEConv(Sequential(Linear(dim, dim), BatchNorm(dim), ReLU(),
                                         Linear(dim, dim), ReLU()),
                                         edge_dim = 11)

        self.norm1 = BatchNorm(dim)
        self.norm2 = BatchNorm(dim)

        l1_size = 512
        input_size = 2*dim*2
        self.mlp = Sequential(Linear(input_size,l1_size),
                                     BatchNorm(l1_size),
                                     ReLU(),
                                     Linear(l1_size, l1_size),
                                     BatchNorm(l1_size),
                                     ReLU(),
                                     Linear(l1_size, l1_size),
                                     BatchNorm(l1_size),
                                     ReLU(),
                                     Linear(l1_size,1))



    def forward(self, x, edge_index, edge_attr, batch):

        atomicNums = self.atomicNumEmb(x[:,0].long())
        chirality = self.chiralityEmb(x[:,1].long())
        formalCharge = self.formalChargeEmb(x[:,2].long())
        hybridizations = self.hybridizationEmb(x[:,4].long())
        numHs = self.numHEmb(x[:,5].long())
        valences = self.valenceEmb(x[:,5].long())
        degrees = self.degreeEmb(x[:,6].long())
        aromatics = self.aromaticEmb(x[:,7].long())
        types = self.typeEmb(x[:,8].long())
        mass = x[:,9].view(-1,1)
        residues = self.residueEmb(x[:,10].long())


        dists = edge_attr[:,0].view(-1,1)
        bondTypes = self.bondTypeEmb(edge_attr[:,1].long())
        bondDirs = self.bondDirEmb(edge_attr[:,2].long())
        stereo = self.stereoEmb(edge_attr[:,3].long())
        inRing = self.inRingEmb(edge_attr[:,4].long())
        inner = self.innerEmb(edge_attr[:,5].long())

        x = torch.cat((atomicNums,chirality,formalCharge,hybridizations,numHs,valences,degrees,aromatics,types,mass, residues),dim=1)
        edge_attr = torch.cat((dists, bondTypes, bondDirs, stereo, inRing, inner), dim=1)
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.norm1(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.norm2(x2)

        x = torch.cat((x1,x2), dim=1)
        x = torch.cat((global_add_pool(x,batch),global_max_pool(x,batch)),dim=1)
        x  = self.mlp(x)
        return x