const mongoose = require('mongoose');

const postSchema = new mongoose.Schema({
  post: {
    type: String,
  },
  owner: {
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User',
    required: true  
  },
  caption: {
    type: String,
  },
  likes: [{
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User'
  }]
}, { timestamps: true }); 


postSchema.index({ owner: 1, createdAt: -1 });

const Post = mongoose.model('Post', postSchema);
module.exports = Post;
