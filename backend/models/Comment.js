const mongoose = require('mongoose');

const commentSchema = new mongoose.Schema({
  postID: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Post',
    required: true   
  },
  comment: {
    type: String,
    required: true 
  },
  owner: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true   
  }
}, { timestamps: true });  

const Comment = mongoose.model('Comment', commentSchema);
module.exports = Comment;
