<template>
  <div
    class="flex flex-col items-center justify-center mt-10 pt-10 border-t border-gray-300"
  >
    <img
      class="w-16 h-16 rounded-full"
      src="~/assets/images/profile.jpg"
      alt="Hugo Matalonga"
    />
    <div class="mt-2 text-center">
      <div class="text-xs text-gray-600 sm:text-sm">Written by</div>
      <div class="text-base text-gray-800 font-semibold sm:text-lg">
        Hugo Matalonga
      </div>
    </div>
    <div class="pb-12 mt-8">
      <nuxt-link
        class="text-sm text-indigo-500 font-semibold hover:text-indigo-600 sm:text-base"
        to="/blog"
        >← View All Posts</nuxt-link
      >
    </div>
  </div>
</template>

<script>
export default {}
</script>

<style></style>
