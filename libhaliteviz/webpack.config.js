const path = require("path");

module.exports = {
    entry: "./src/main.js",
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "bundle.js",
        library: "libhaliteviz",
        libraryTarget: "window",
        publicPath: "dist/",
    },
    devtool: "source-map",
    resolve: {
        alias: {
            "path": require.resolve("path-browserify"),
            "url": require.resolve("url/"),
        },
    },
    module: {
        noParse: /libzstd/,
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                },
            },
            {
                test: /\.png$/,
                loader: "file-loader",
            },
        ],
    },
};
