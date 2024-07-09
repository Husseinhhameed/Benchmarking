class DatasetGenerator(Dataset):
    def __init__(self, dataset_dir, img_width, img_height, batch_size, shuffle_sessions=True):
        self.dataset_dir = dataset_dir
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.shuffle_sessions = shuffle_sessions
        self.data = []
        self.sessions = [os.path.join(dataset_dir, session) for session in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, session))]

        if shuffle_sessions:
            np.random.shuffle(self.sessions)

        for session in self.sessions:
            self._load_session_data(session)

        # Global shuffle of all data points after initial loading
        np.random.shuffle(self.data)

        self.on_epoch_end()  # Prepare for the first epoch

    def _load_session_data(self, session_dir):
        snapshots_dir = os.path.join(session_dir, 'Snapshots')
        depthmap_dir = os.path.join(session_dir, 'DepthMaps')
        camera_positions_file = os.path.join(snapshots_dir, 'CameraPositions.csv')

        session_data = []  # Temporary list to hold session data

        try:
            df = pd.read_csv(camera_positions_file)
        except Exception as e:
            print(f"Error loading camera positions from {camera_positions_file}: {e}")
            return

        session_name = os.path.basename(session_dir)

        for _, row in df.iterrows():
            timestamp = str(row['Timestamp']).replace('_', '')
            rgb_image_name = f"Snapshot_{row['Timestamp']}.png"
            depth_image_name = f"DepthMap{timestamp}.png"

            rgb_image_path = os.path.join(snapshots_dir, rgb_image_name)
            depth_image_path = os.path.join(depthmap_dir, depth_image_name)

            if os.path.exists(rgb_image_path) and os.path.exists(depth_image_path):
                pose_data = row[['DeltaPositionX', 'DeltaPositionY', 'DeltaPositionZ',
                                 'DeltaRotationX', 'DeltaRotationY', 'DeltaRotationZ', 'DeltaRotationW']].to_numpy()
                session_data.append(( rgb_image_path, depth_image_path, pose_data))

        # Shuffle data points within this session
        np.random.shuffle(session_data)
        self.data.extend(session_data)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        rgb_image_path, depth_image_path, pose_data = self.data[idx]

        # Load and preprocess the RGB image
        rgb_image = cv2.imread(rgb_image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # Create a CLAHE object (with specified clip limit and tile grid size)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):  # Apply CLAHE to each channel of the RGB image
            rgb_image[:, :, i] = clahe.apply(rgb_image[:, :, i])

        # Normalize the RGB image to 0-1
        rgb_image = rgb_image / 255.0
        rgb_image_tensor = ToTensor()(rgb_image).float()  # Convert to FloatTensor

        # Load and preprocess the depth map
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image.ndim > 2:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        depth_image = depth_image.astype(np.float32) / 255.0  # Normalize depth images
        depth_map_tensor = ToTensor()(depth_image).float()  # Convert to FloatTensor


        # Process pose data
        pose_data = np.array(pose_data, dtype=np.float32)
        pose_data_tensor = torch.tensor(pose_data, dtype=torch.float)  # Ensure pose data is FloatTensor

        return rgb_image_tensor, depth_map_tensor, pose_data_tensor



    def on_epoch_end(self):
        if self.shuffle_sessions:
            np.random.shuffle(self.sessions)
            self.data = []  # Clear existing data
            for session in self.sessions:
                self._load_session_data(session)
            # Global shuffle of all data points after reloading
            np.random.shuffle(self.data)
