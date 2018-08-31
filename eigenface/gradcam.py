def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def grad_cam(input_model, image, class_score, category_index, layer_name):
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)

    loss = K.sum(model.layers[-1].output)
    # conv_output = [l for l in model.layers[0].layers if l.name is layer_name][0].output
    # conv_output = [l for l in model.layers if l.name is layer_name][0].output
    conv_output = model.get_layer(layer_name).output

    K.set_learning_phase(0)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    # gradient_function = K.function([model.layers[0].input], [conv_output])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    # output = gradient_function([image])
    # output = output[0,:]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.zeros(image.shape[1: 3], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * cv2.resize(output[:, :, i], (224, 224), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(cam, cmap = 'gray')
        # plt.show()
    # print weights[0]

    # cam = cv2.resize(cam, (224, 224), interpolation = cv2.INTER_CUBIC)
    cam = np.maximum(cam, 0)
    # print np.max(cam)
    heatmap = cam * 1.0 / np.max(cam)
    cam[0, 0] = np.max(cam) * 1.0 / class_score
    cam = cam * 1.0 / np.max(cam)
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    # Return to BGR [0..255] from the preprocessed image
    # image = image[0, :]
    # image -= np.min(image)
    # image = np.minimum(image, 255)

    # cam = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    # cam = np.float32(cam) + np.float32(image)
    # cam = 255 * cam / np.max(cam)
    # cam = cam.astype(np.int8)
    # return np.uint8(cam), heatmap
    return cam, heatmap, output, grads_val